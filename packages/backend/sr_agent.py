#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Literal, TypedDict, Dict, Any, Optional, Tuple, AsyncGenerator

from dotenv import load_dotenv
from SRQuestionGenerator import QuestionGenerator
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
# from postgres import AsyncPostgresSaver  # Lazily imported in _initialize_agent to avoid libpq issues in tests
from parsing import parse_langgraph_stream
from protocol import StreamProtocolPart
from agent_state import State, get_state_from_supabase
from chat.ChatSession import ChatSession
import uuid
from supabase import Client, create_client

from VideoClip import VideoClip


load_dotenv()


# Validate environment variables (mirror esi_agent.py pattern)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
database_url = os.getenv("DATABASE_URL")

if not all([supabase_url, supabase_key, database_url]):
    raise ValueError(
        "Required environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL"
    )

supabase: Client = create_client(supabase_url, supabase_key)


class MediaPart(TypedDict):
    type: Literal["media"]
    mime_type: Literal["video/mp4"]
    data: str


def prepare_video_context(clip_uuids: List[str]) -> List[MediaPart]:
    """Prepare base64-encoded media parts for the given clip UUIDs.

    Mirrors esi_agent.prepare_video_context but tailored for SR usage.
    """
    media_parts: List[MediaPart] = []

    for uuid_val in clip_uuids:
        try:
            clip = VideoClip(uuid_val)
            base64_data = clip.download_base64()
            media_parts.append(
                {
                    "type": "media",
                    "mime_type": "video/mp4",
                    "data": base64_data,
                }
            )
        except Exception as e:
            try:
                print(f"Failed to prepare video {uuid_val}: {e}")
            except Exception:
                pass
            continue

    return media_parts


# ---------------------------------------------------------------------
# Phase 1: AgentGraphState schema and defaults (JSON-friendly)
# ---------------------------------------------------------------------

class QuestionState(TypedDict):
    """Serializable cue-based question."""

    text_cue: str
    answer: str


class QueueItemState(TypedDict):
    """One scheduled spaced-retrieval task for a clip."""

    clip_id: str
    next_at: str  # ISO-8601
    interval_index: int
    attempt_count: int
    questions: List[QuestionState]
    meta: Dict[str, Any]


class ActiveSessionExchange(TypedDict, total=False):
    question: str
    user: str
    assessment: str


class ActiveSessionState(TypedDict):
    """Ongoing Q&A session state."""

    clip_id: str
    q_index: int
    exchanges: List[ActiveSessionExchange]
    interval_index: int
    questions: List[QuestionState]
    attempt_count: int


class SRState(TypedDict, total=False):
    """Top-level SR slice written into AgentGraphState."""

    interval_seconds: List[int]
    questions_per_clip: int
    candidate_limit: int
    max_clips: int
    selected_clips: List[str]
    enqueued_clips: List[str]
    clip_questions: Dict[str, List[QuestionState]]
    queue: List[QueueItemState]
    active_session: Optional[ActiveSessionState]
    last_activity_at: str
    errors: List[str]
    # New fields for stage-aware orchestration
    stage: Literal["kickoff", "session_active", "feedback_next", "waiting", "idle"]
    last_stage_change_at: str
    media_attached_clips: List[str]
    pending_eval_q_index: Optional[int]


class AgentGraphState(TypedDict, total=False):
    sr: SRState


# Defaults and configurable keys
DEFAULT_INTERVAL_SECONDS: List[int] = [30, 60, 120, 240]
DEFAULT_QUESTIONS_PER_CLIP: int = 4
DEFAULT_CANDIDATE_LIMIT: int = 20
DEFAULT_VIDEO_TABLE: str = "videos"
DEFAULT_VIDEO_TS_COLUMN: str = "time_created"


def ensure_sr_slice(state: Dict[str, Any]) -> SRState:
    """Ensure SR slice exists in AgentGraphState with sane defaults.

    Returns the writable SRState dict contained within the provided state.
    """
    if "sr" not in state or not isinstance(state.get("sr"), dict):
        state["sr"] = {}
    sr: SRState = state["sr"]  # type: ignore[assignment]

    # Initialize defaults if missing
    if not isinstance(sr.get("interval_seconds"), list):
        sr["interval_seconds"] = list(DEFAULT_INTERVAL_SECONDS)
    if not isinstance(sr.get("questions_per_clip"), int):
        sr["questions_per_clip"] = DEFAULT_QUESTIONS_PER_CLIP
    if not isinstance(sr.get("candidate_limit"), int):
        sr["candidate_limit"] = DEFAULT_CANDIDATE_LIMIT
    if not isinstance(sr.get("selected_clips"), list):
        sr["selected_clips"] = []
    if not isinstance(sr.get("enqueued_clips"), list):
        sr["enqueued_clips"] = []
    if not isinstance(sr.get("clip_questions"), dict):
        sr["clip_questions"] = {}
    if not isinstance(sr.get("queue"), list):
        sr["queue"] = []
    if "active_session" not in sr:
        sr["active_session"] = None
    if not isinstance(sr.get("errors"), list):
        sr["errors"] = []
    # Initialize new stage-related fields
    if sr.get("stage") not in {"kickoff", "session_active", "feedback_next", "waiting", "idle"}:
        sr["stage"] = "idle"
    if not isinstance(sr.get("last_stage_change_at"), str):
        sr["last_stage_change_at"] = _to_iso(datetime.utcnow())
    if not isinstance(sr.get("media_attached_clips"), list):
        sr["media_attached_clips"] = []
    if "pending_eval_q_index" not in sr:
        sr["pending_eval_q_index"] = None

    return sr


def get_stage(state: Dict[str, Any]) -> str:
    sr = ensure_sr_slice(state)
    return str(sr.get("stage", "idle"))


def set_stage(state: Dict[str, Any], stage: str) -> bool:
    """Set stage if changed; update last_stage_change_at. Returns True if changed."""
    sr = ensure_sr_slice(state)
    current = str(sr.get("stage", "idle"))
    if current == stage:
        return False
    sr["stage"] = stage
    sr["last_stage_change_at"] = _to_iso(datetime.utcnow())
    return True


def _parse_int_list_env(var_name: str, fallback: List[int]) -> List[int]:
    raw = os.getenv(var_name)
    if not raw:
        return list(fallback)
    try:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        vals = [int(p) for p in parts]
        return vals or list(fallback)
    except Exception:
        return list(fallback)


def _parse_int_env(var_name: str, fallback: int) -> int:
    raw = os.getenv(var_name)
    if not raw:
        return int(fallback)
    try:
        val = int(raw)
        return val if val > 0 else int(fallback)
    except Exception:
        return int(fallback)


def configure_sr_from_env(state: Dict[str, Any]) -> None:
    """Apply environment overrides to the SR slice (intervals, counts, limits).

    This is idempotent and safe to call at kickoff/chat boundaries.
    """
    sr = ensure_sr_slice(state)

    intervals = _parse_int_list_env("SR_INTERVAL_SECONDS", sr.get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
    intervals = [x for x in intervals if isinstance(x, int) and x > 0]
    sr["interval_seconds"] = intervals or list(DEFAULT_INTERVAL_SECONDS)

    sr["questions_per_clip"] = _parse_int_env("SR_QUESTIONS_PER_CLIP", sr.get("questions_per_clip", DEFAULT_QUESTIONS_PER_CLIP))
    sr["candidate_limit"] = _parse_int_env("SR_CANDIDATE_LIMIT", sr.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT))
    # number of clips per SR run
    sr["max_clips"] = _parse_int_env("SR_MAX_CLIPS", int(sr.get("max_clips", 3)))

    # Video source configuration is read where discovery is performed; export defaults via env
    sr.setdefault("video_table", os.getenv("SR_VIDEO_TABLE", DEFAULT_VIDEO_TABLE))
    sr.setdefault("video_ts_column", os.getenv("SR_VIDEO_TS_COLUMN", DEFAULT_VIDEO_TS_COLUMN))


def _log_sr_error(state: Dict[str, Any], msg: str, exc: Optional[BaseException] = None) -> None:
    """Print and persist a timestamped SR error message in state."""
    ts = datetime.utcnow().isoformat()
    full = f"[SR][{ts}] {msg}"
    try:
        if exc is not None:
            full = f"{full} | {type(exc).__name__}: {exc}"
        print(full)
    except Exception:
        pass
    sr = ensure_sr_slice(state)
    errs = sr.get("errors", []) or []
    errs.append(full)
    # Keep last 50 for brevity
    sr["errors"] = errs[-50:]


def _extract_latest_sr_state_from_messages(messages: List[Any]) -> Optional[SRState]:
    """Scan messages from newest to oldest for a SystemMessage named 'sr_state' and parse JSON content."""
    for m in reversed(messages or []):
        try:
            if m.__class__.__name__ == "SystemMessage" and getattr(m, "name", None) == "sr_state":
                raw = getattr(m, "content", "")
                data = json.loads(raw) if isinstance(raw, str) else None
                if isinstance(data, dict):
                    # Coerce to SRState shape
                    sr_state: SRState = data  # type: ignore[assignment]
                    return sr_state
        except Exception:
            continue
    return None


def _persist_chat_message(session_id: str, role: str, message_obj: Any) -> None:
    try:
        supabase.table("chat_messages").insert(
            {
                "role": role,
                "data": message_obj.__dict__,
                "session_id": str(session_id),
            }
        ).execute()
    except Exception:
        pass


def _append_and_persist_sr_state(session_id: str, state: Dict[str, Any]) -> None:
    sr = ensure_sr_slice(state)
    try:
        payload = json.dumps(sr)
    except Exception:
        payload = json.dumps({})
    sr_msg = SystemMessage(content=payload, name="sr_state")
    # Append into current messages and persist to supabase
    msgs = state.get("messages", []) or []
    msgs.append(sr_msg)
    state["messages"] = msgs
    _persist_chat_message(session_id, "system", sr_msg)


# ---------------------------------------------------------------------
# Phase 2: Pure state helpers (discovery, selection, scheduling, sessions)
# ---------------------------------------------------------------------

def _to_naive_utc(dt: datetime) -> datetime:
    if getattr(dt, "tzinfo", None) is not None:
        try:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            return datetime.utcnow()
    return dt


def _to_iso(dt: datetime) -> str:
    return _to_naive_utc(dt).isoformat()


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return _to_naive_utc(dt)
    except Exception:
        return None


def _dedupe_preserve_order(ids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in ids:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def load_recent_video_ids(state: Dict[str, Any]) -> List[str]:
    """Load recent video UUIDs from Supabase (videos table).

    Orders by configured timestamp column desc; falls back to `created_at`.
    Respects `sr.candidate_limit`.
    """
    sr = ensure_sr_slice(state)
    table = str(sr.get("video_table", DEFAULT_VIDEO_TABLE))
    ts_col = str(sr.get("video_ts_column", DEFAULT_VIDEO_TS_COLUMN))
    limit = int(sr.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT))

    try:
        # Preferred ordering by configured timestamp column
        query = supabase.table(table).select(f"id, {ts_col}").order(ts_col, desc=True)
        if limit:
            query = query.limit(limit)
        resp = query.execute()
        data = resp.data or []
        ids = [str(row["id"]) for row in data if row.get("id")]
        if ids:
            return _dedupe_preserve_order(ids)
    except Exception as e:
        _log_sr_error(state, f"Failed to load recent video ids ordered by {ts_col}", e)

    # Fallback ordering by created_at
    try:
        query = supabase.table(table).select("id, created_at").order("created_at", desc=True)
        if limit:
            query = query.limit(limit)
        resp = query.execute()
        data = resp.data or []
        ids = [str(row["id"]) for row in data if row.get("id")]
        return _dedupe_preserve_order(ids)
    except Exception as e:
        _log_sr_error(state, "Failed to load recent video ids by created_at", e)
        return []


def select_first_n(candidates: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    return list(_dedupe_preserve_order(candidates)[:n])


def enqueue(
    state: Dict[str, Any],
    *,
    clip_id: str,
    questions: List[QuestionState],
    interval_index: int,
    base_time: Optional[datetime] = None,
    attempt_count: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    sr = ensure_sr_slice(state)
    intervals = sr.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)
    if interval_index < 0 or interval_index >= len(intervals):
        raise ValueError("interval_index out of range")
    reference = _to_naive_utc(base_time or datetime.utcnow())
    next_at = reference + timedelta(seconds=int(intervals[interval_index]))
    item: QueueItemState = {
        "clip_id": clip_id,
        "next_at": _to_iso(next_at),
        "interval_index": int(interval_index),
        "attempt_count": int(attempt_count) if attempt_count >= 0 else 0,
        "questions": list(questions or []),
        "meta": dict(meta) if meta else {},
    }
    q = sr.get("queue", []) or []
    q.append(item)
    sr["queue"] = q


def _head_due_index(q: List[QueueItemState], now_dt: datetime) -> Optional[int]:
    best_idx: Optional[int] = None
    best_time: Optional[datetime] = None
    for idx, it in enumerate(q):
        dt = _parse_iso(it.get("next_at", ""))
        if dt is None:
            continue
        if now_dt >= dt:
            if best_time is None or dt < best_time:
                best_time = dt
                best_idx = idx
    return best_idx


def get_next_due(state: Dict[str, Any], now: Optional[datetime] = None) -> Optional[QueueItemState]:
    sr = ensure_sr_slice(state)
    q = sr.get("queue", []) or []
    now_dt = _to_naive_utc(now or datetime.utcnow())
    idx = _head_due_index(q, now_dt)
    return q[idx] if idx is not None else None


def pop_next_due(state: Dict[str, Any], now: Optional[datetime] = None) -> Optional[QueueItemState]:
    sr = ensure_sr_slice(state)
    q = sr.get("queue", []) or []
    now_dt = _to_naive_utc(now or datetime.utcnow())
    idx = _head_due_index(q, now_dt)
    if idx is None:
        return None
    item = q.pop(idx)
    sr["queue"] = q
    return item


def begin_session(state: Dict[str, Any], item: QueueItemState) -> str:
    """Create an active session from a queue item and return first prompt."""
    sr = ensure_sr_slice(state)
    questions = item.get("questions", []) or []
    sr["active_session"] = {
        "clip_id": item.get("clip_id", ""),
        "q_index": 0,
        "exchanges": [],
        "interval_index": int(item.get("interval_index", 0)),
        "questions": list(questions),
        "attempt_count": int(item.get("attempt_count", 0)),
    }
    return current_prompt(state) or "Take a moment to recall one detail from this clip."


def current_prompt(state: Dict[str, Any]) -> Optional[str]:
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return None
    qs = sess.get("questions", []) or []
    q_index = int(sess.get("q_index", 0))
    if q_index < 0 or q_index >= len(qs):
        return None
    text = str(qs[q_index].get("text_cue", "")).strip()
    return text or None


def append_answer_and_advance(state: Dict[str, Any], answer: str) -> None:
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return
    qs = sess.get("questions", []) or []
    q_index = int(sess.get("q_index", 0))
    if q_index < 0 or q_index >= len(qs):
        return
    question_text = str(qs[q_index].get("text_cue", ""))
    expected_answer = str(qs[q_index].get("answer", ""))
    exchanges = list(sess.get("exchanges", []))
    exchanges.append({
        "question": question_text,
        "user": str(answer or "").strip(),
        "assessment": "",
        "q_index": int(q_index),
        "expected_answer": expected_answer,
    })
    sess["exchanges"] = exchanges
    sess["q_index"] = q_index + 1
    sr["active_session"] = sess


def session_finished(state: Dict[str, Any]) -> bool:
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return True
    qs = sess.get("questions", []) or []
    q_index = int(sess.get("q_index", 0))
    return q_index >= len(qs)


def evaluate_session(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return {score: float, success: bool} prioritizing assessment-based correctness.

    If per-exchange "assessment" exists (correct/incorrect), use it.
    Otherwise, fallback to counting non-empty answers as correct.
    Success threshold: 0.7.
    """
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return {"score": 0.0, "success": False}
    qs = sess.get("questions", []) or []
    exchanges = list(sess.get("exchanges", []))
    total = max(1, len(qs))
    have_assessment = any(
        isinstance(ex, dict) and ex.get("assessment") in {"correct", "incorrect"} for ex in exchanges
    )
    correct_count = 0
    if have_assessment:
        for ex in exchanges:
            try:
                if ex.get("assessment") == "correct":
                    correct_count += 1
            except Exception:
                continue
    else:
        for ex in exchanges:
            try:
                if str(ex.get("user", "")).strip():
                    correct_count += 1
            except Exception:
                continue
    score = min(1.0, correct_count / float(total))
    return {"score": score, "success": score >= 0.7}


def _soft_match_correct(user_answer: str, expected_answer: str) -> bool:
    ua = (user_answer or "").strip().lower()
    ea = (expected_answer or "").strip().lower()
    if not ua or not ea:
        return False
    # simple synonyms/variants
    synonyms = {
        "splash": {"splash", "splashing", "splashing around", "splash around", "playing in water"},
        "black": {"black", "dark", "dark-colored", "dark coloured"},
        "canyon": {"canyon", "canyons", "rocky canyon", "rocky canyons", "river canyon", "rocky valley"},
        "valley": {"valley", "river valley", "rocky valley"},
        "two": {"two", "2", "a couple", "couple"},
    }
    for key, variants in synonyms.items():
        if key in ea:
            if any(v in ua for v in variants):
                return True
    # token overlap heuristic
    ua_tokens = set(ua.replace("\n", " ").split())
    ea_tokens = set(ea.replace("\n", " ").split())
    if len(ea_tokens) > 0 and len(ua_tokens & ea_tokens) / len(ea_tokens) >= 0.45:
        return True
    # substring
    if ua in ea or ea in ua:
        return True
    return False


def _evaluate_answer_with_llm(question_text: str, expected_answer: str, user_answer: str) -> Dict[str, Any]:
    """Use the chat model to judge if the user's answer is correct.

    Returns a dict: {"correct": bool, "feedback": str, "correct_answer": str}
    """
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    instruction = (
        "You are grading a short recall answer for spaced retrieval. Decide if the user's answer matches the expected answer. "
        "Be lenient: accept paraphrases, synonyms, plural/singular variants, close color/number descriptors, and everyday phrasing. "
        "Favor semantic equivalence over exact wording; only mark incorrect if clearly incompatible. If in doubt, mark correct. "
        "Return ONLY JSON: {\"correct\": true|false, \"feedback\": string, \"correct_answer\": string}. Keep feedback one short sentence.\n\n"
        "Examples (grade as correct):\n"
        "- expected: 'splashing in the river'; user: 'they were splashing around'\n"
        "- expected: 'rocky canyons'; user: 'rocky canyon'\n"
        "- expected: 'black shorts'; user: 'dark shorts'\n"
        "- expected: 'two people'; user: 'a couple'\n"
    )
    payload = {
        "question": question_text,
        "expected_answer": expected_answer,
        "user_answer": user_answer,
    }
    resp = model.invoke(
        [
            SystemMessage(content=instruction),
            HumanMessage(content=f"Data:\n{json.dumps(payload, ensure_ascii=False)}"),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else getattr(resp, "content", "")
    result = {"correct": False, "feedback": "", "correct_answer": expected_answer}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed_correct = bool(parsed.get("correct", False))
            # If model says incorrect but soft match thinks it's correct, flip to correct
            if not parsed_correct and _soft_match_correct(user_answer, expected_answer):
                parsed_correct = True
                parsed["feedback"] = parsed.get("feedback") or "Correct."
            result["correct"] = parsed_correct
            result["feedback"] = str(parsed.get("feedback", "")).strip() or ("Correct." if parsed_correct else "That's not quite right.")
            result["correct_answer"] = str(parsed.get("correct_answer", expected_answer))
            return result
    except Exception:
        # salvage JSON if wrapped
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    parsed_correct = bool(parsed.get("correct", False))
                    if not parsed_correct and _soft_match_correct(user_answer, expected_answer):
                        parsed_correct = True
                        parsed["feedback"] = parsed.get("feedback") or "Correct."
                    result["correct"] = parsed_correct
                    result["feedback"] = str(parsed.get("feedback", "")).strip() or ("Correct." if parsed_correct else "That's not quite right.")
                    result["correct_answer"] = str(parsed.get("correct_answer", expected_answer))
                    return result
        except Exception:
            pass
    # fallback heuristic (soft match)
    if _soft_match_correct(user_answer, expected_answer):
        return {"correct": True, "feedback": "Correct.", "correct_answer": expected_answer}
    return {"correct": False, "feedback": "That's not quite right.", "correct_answer": expected_answer}


def reschedule(state: Dict[str, Any], success: bool, now: Optional[datetime] = None) -> None:
    """Reschedule the active session clip at the next/previous interval and clear session."""
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return
    next_idx: int
    intervals = sr.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)
    last_index = len(intervals) - 1
    cur_idx = int(sess.get("interval_index", 0))
    if success:
        next_idx = min(cur_idx + 1, last_index)
    else:
        next_idx = max(cur_idx - 1, 0)
    attempt = int(sess.get("attempt_count", 0)) + 1
    clip_id = str(sess.get("clip_id", ""))
    questions = list(sess.get("questions", []))
    enqueue(state, clip_id=clip_id, questions=questions, interval_index=next_idx, base_time=now or datetime.utcnow(), attempt_count=attempt)
    sr["active_session"] = None
    # Make stage explicit for idle/waiting state
    try:
        set_stage(state, "waiting")
    except Exception:
        pass


# ---------------------------------------------------------------------
# Runner (LangGraph agent) - mirrors esi_agent.py
# ---------------------------------------------------------------------


def _build_system_prompt_master() -> str:
    return (
        "ROLE\n"
        "You are a supportive Spaced Retrieval (SR) memory coach helping a person practice recall from first-person smart‑glasses clips. "
        "You run inside a stateful LangGraph agent whose state persists between turns. Your job is to strengthen recall using short, concrete questions at increasing intervals.\n\n"

        "END‑TO‑END FLOW (HIGH LEVEL)\n"
        "1) Kickoff: select a small subset of clips and ask once if they are ready to begin.\n"
        "2) Session: for the current clip, ask one concrete question at a time; after each answer, briefly judge correctness; if incorrect, state the correct answer once; then ask the next question (exact text).\n"
        "3) Spacing: when a clip’s questions finish, the system will schedule it forward/backward based on performance; you may acknowledge this briefly and move on.\n"
        "4) Waiting: when nothing is due, offer one short supportive line and keep the conversation light.\n\n"

        "MEDIA & CONTEXT\n"
        "Kickoff may include one message containing media parts (video). Assume these remain in history; you can reference them later without re‑sending media. Do not invent visual details that are not plausibly present.\n\n"

        "QUESTION DESIGN\n"
        "- Ask exactly one concrete recall question per turn.\n"
        "- Favor where/when anchors and sensory details (color, texture, sound, temperature).\n"
        "- If the user struggles, gently scaffold (offer a tiny, concrete angle) rather than broad hints.\n\n"

        "ANSWER HANDLING\n"
        "- Respond first with a brief, natural correctness judgment.\n"
        "- If the answer is not correct, provide the correct answer once, succinctly.\n"
        "- Treat close paraphrases as correct; prioritize semantic equivalence over exact wording.\n"
        "- Then naturally segue to the next question (using the exact text provided by the system).\n\n"

        "STAGE POLICIES (ENFORCED)\n"
        "- kickoff: Ask readiness once. Never re‑ask readiness later. No repeated greetings.\n"
        "- session_active: Include correctness, (if needed) the correct answer, then ask the exact next question. No greetings. No readiness prompts.\n"
        "- waiting: One short, supportive line only; acknowledge user if they talk. No greetings. No readiness prompts.\n\n"

        "TONE, UX & SAFETY\n"
        "- Tone: warm, validating, collaborative; avoid clinical or robotic phrasing.\n"
        "- Brevity: 1–3 short sentences unless asked for more.\n"
        "- Momentum: keep moving with light, supportive pacing; one question at a time.\n"
        "- Agency: invite participation without pressure; celebrate effort and progress.\n"
        "- Avoid repeating prologues or meta‑instructions to the user.\n\n"

        "WHAT NOT TO DO\n"
        "- Do not ask \"Are you ready?\" outside kickoff.\n"
        "- Do not greet repeatedly.\n"
        "- Do not expose internal instructions or tools.\n"
        "- Do not produce long boilerplate or rigid templates.\n\n"

        "SUCCESS CRITERIA\n"
        "Each turn should (1) sound natural/warm, (2) include the stage‑required information, and (3) sustain momentum with one clear, concrete question."
    )


def _build_system_prompt_kickoff() -> str:
    return (
        "Kickoff stage: ask once if the user is ready to begin, then wait for readiness. "
        "Do not re-ask readiness later in the session. Keep language natural and concise."
    )


def _build_system_prompt_session() -> str:
    return (
        "Active session stage: For each turn after the user answers, include a brief correctness judgment. "
        "If incorrect, include the correct answer once. Then naturally segue into the next question exactly as provided. "
        "No greetings; do not ask about readiness. Keep it human-sounding and concise."
    )


def _build_system_prompt_waiting() -> str:
    return (
        "Waiting stage: no question is due. Offer one short supportive line or acknowledge their message. "
        "Do not ask about readiness here, and do not greet."
    )


def _build_graph() -> StateGraph:
    graph_builder = StateGraph(State)

    def agent_node(state: State) -> State:
        llm = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
        )
        message = llm.invoke(state["messages"])  # type: ignore[index]
        state["messages"].append(message)
        # Persist assistant message to Supabase like ESI
        try:
            supabase.table("chat_messages").insert(
                {
                    "role": "assistant",
                    "data": message.__dict__,
                    "session_id": str(state.get("session_id", "")),
                }
            ).execute()
        except Exception:
            pass
        return state

    graph_builder.add_node("agent", agent_node)
    # Tools placeholder for future HUD/tool calls
    tools_node = ToolNode(tools=[])
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()

async def kickoff(session_id: str) -> AsyncGenerator[StreamProtocolPart, None]:
    graph = _build_graph()
    # Create ChatSession row (mirror ESI)
    ChatSession(session_id=str(session_id)).save_to_supabase()
    # Build initial state container
    state: Dict[str, Any] = {"messages": [], "session_id": str(session_id)}
    # Initialize config/defaults on sr slice
    configure_sr_from_env(state)

    # Discover and select
    candidates = load_recent_video_ids(state)
    if not candidates:
        fb = (os.getenv("SR_FALLBACK_CLIP_IDS") or "").strip()
        if fb:
            candidates = _dedupe_preserve_order([cid.strip() for cid in fb.split(",") if cid.strip()])
        if not candidates:
            _log_sr_error(state, "No candidate clips found (Supabase and fallback empty)")
    sr = ensure_sr_slice(state)
    max_clips = int(sr.get("max_clips", 3))
    selected = select_first_n(candidates, n=max_clips)
    sr["selected_clips"] = list(selected)
    sr["enqueued_clips"] = []

    # Prepare questions and enqueue
    qpc = int(sr.get("questions_per_clip", DEFAULT_QUESTIONS_PER_CLIP))
    now = datetime.utcnow()
    for cid in selected:
        try:
            qgen = QuestionGenerator(VideoClip(cid))
            questions_objs = qgen.generate(num_questions=qpc) or []
            questions: List[QuestionState] = [
                {"text_cue": str(q.text_cue), "answer": str(q.answer)} for q in questions_objs
            ]
            if not questions:
                _log_sr_error(state, f"No questions generated for clip {cid}")
                continue
            cq = sr.get("clip_questions", {}) or {}
            cq[cid] = questions
            sr["clip_questions"] = cq
            enqueue(state, clip_id=cid, questions=questions, interval_index=0, base_time=now, attempt_count=0)
            sr["enqueued_clips"].append(cid)
        except Exception as e:
            _log_sr_error(state, f"Failed to prepare questions/enqueue for {cid}", e)
            continue

    # Prepare media context for all selected clips (attach once at kickoff)
    media_parts_all: List[MediaPart] = []
    if selected:
        try:
            media_parts_all = prepare_video_context(selected)
        except Exception as e:
            _log_sr_error(state, "Failed to prepare media context for selected clips", e)
    sr["media_attached_clips"] = list(selected)

    # Fast-start vs readiness-gated kickoff (default: readiness-gated)
    first_prompt: Optional[str] = None
    fast_start_env = (os.getenv("SR_FAST_START", "0").strip().lower() in {"1", "true", "yes"})
    if fast_start_env:
        try:
            q = sr.get("queue", []) or []
            earliest_idx: Optional[int] = None
            earliest_time: Optional[datetime] = None
            for idx, it in enumerate(q):
                dt = _parse_iso(it.get("next_at", "")) or now
                if earliest_time is None or dt < earliest_time:
                    earliest_time = dt
                    earliest_idx = idx
            if earliest_idx is not None:
                item = q.pop(earliest_idx)
                sr["queue"] = q
                first_prompt = begin_session(state, item)
                set_stage(state, "session_active")
            else:
                set_stage(state, "kickoff")
        except Exception as e:
            _log_sr_error(state, "Fast-start selection failed; falling back to readiness prompt", e)
            set_stage(state, "kickoff")
    else:
        set_stage(state, "kickoff")

    # Build streaming payload
    master_prompt = _build_system_prompt_master()
    if first_prompt:
        human_content: List[Any] = [
            {"type": "text", "text": f"Ask exactly this question to begin: {first_prompt}"},
        ]
    else:
        n = len(sr.get("enqueued_clips", []))
        if n == 0:
            _log_sr_error(state, "Selected clips, but none enqueued (no questions)")
        human_content = [
            {"type": "text", "text": f"We will practice recall for {n} short clips. Ask exactly: Are you ready to begin?"},
        ]
    stage_prompt = _build_system_prompt_kickoff()
    state["messages"] = [
        SystemMessage(content=master_prompt),
        SystemMessage(content=stage_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": "Context: media for all selected clips (reference these clips during this SR session)."},
                *media_parts_all,
            ]
        ),
        HumanMessage(content=human_content),
    ]
    # Persist kickoff messages to Supabase
    for m in state["messages"]:
        role = "system" if isinstance(m, SystemMessage) else "user"
        _persist_chat_message(session_id, role, m)
    # Persist SR slice message
    _append_and_persist_sr_state(session_id, state)

    # Stream with graph
    async for part in parse_langgraph_stream(
        graph.astream(state, stream_mode="messages")
    ):
        yield part

async def chat(session_id: str, user_message: str) -> AsyncGenerator[StreamProtocolPart, None]:
    graph = _build_graph()
    # Load existing message history from Supabase
    state: State = get_state_from_supabase(session_id)
    # Restore SR slice from latest persisted sr_state system message
    latest_sr = _extract_latest_sr_state_from_messages(state.get("messages", []))  # type: ignore[index]
    if latest_sr:
        state["sr"] = latest_sr
    sr = ensure_sr_slice(state)  # type: ignore[arg-type]
    sr["last_activity_at"] = _to_iso(datetime.utcnow())

        # If session active: record answer and determine next step
        sess = sr.get("active_session")
        human_content: List[Any]
        # Always append the user's utterance to history first to preserve continuity
        prior_messages = state.get("messages", []) or []  # type: ignore[index]
        user_text = (user_message or "")
        if user_text:
            um = HumanMessage(content=user_text)
            prior_messages = prior_messages + [um]
            _persist_chat_message(session_id, "user", um)
        if isinstance(sess, dict):
            # On transition into session_active, append stage system prompt once
            if set_stage(state, "session_active"):
                sp = SystemMessage(content=_build_system_prompt_session())
                prior_messages.append(sp)
                _persist_chat_message(session_id, "system", sp)
            # Evaluate user's answer with LLM against expected answer for the current question
            qs = sess.get("questions", []) or []
            q_index = int(sess.get("q_index", 0))
            # Snapshot pending eval index so we grade against the intended question
            sr["pending_eval_q_index"] = q_index
            expected_answer = ""
            question_text = ""
            if 0 <= q_index < len(qs):
                expected_answer = str(qs[q_index].get("answer", ""))
                question_text = str(qs[q_index].get("text_cue", ""))
            eval_res = _evaluate_answer_with_llm(question_text, expected_answer, user_message)

            # Record exchange with index/expected answer and advance
            append_answer_and_advance(state, user_message)
            # Patch the exchange that matches the pending index with assessment
            sess2 = ensure_sr_slice(state).get("active_session")
            if isinstance(sess2, dict):
                exchanges = list(sess2.get("exchanges", []))
                target_idx = sr.get("pending_eval_q_index")
                for ex in reversed(exchanges):
                    if isinstance(ex, dict) and ex.get("q_index") == target_idx:
                        ex["assessment"] = "correct" if eval_res.get("correct") else "incorrect"
                        break
                sess2["exchanges"] = exchanges
                ensure_sr_slice(state)["active_session"] = sess2
            sr["pending_eval_q_index"] = None

            feedback_lines = []
            if eval_res.get("correct"):
                feedback_lines.append("Correct.")
            else:
                ca = str(eval_res.get("correct_answer", "")).strip()
                feedback_lines.append("That's not quite right.")
                if ca:
                    feedback_lines.append(f"Correct answer: {ca}")

            if not session_finished(state):
                prompt = current_prompt(state) or "Notice one concrete detail you remember from this clip."
                clip_id = str(sess.get("clip_id", ""))
                # Do not reattach media; it was attached once at kickoff
                human_content = [
                    {"type": "text", "text": " ".join(feedback_lines) + f"\nNext question (ask exactly as written): {prompt}"},
                ]
            else:
                result = evaluate_session(state)
                reschedule(state, success=bool(result.get("success", False)), now=datetime.utcnow())
                summary = "Session complete. " + ("Strong recall." if result.get("success") else "We will revisit soon to reinforce.")
                human_content = [
                    {"type": "text", "text": summary},
                ]
        else:
            # No active session
            now = datetime.utcnow()
            user_text_norm = (user_message or "").strip().lower()
            if get_stage(state) == "kickoff":
                # Gate on readiness (treat any substantive response as readiness too)
                substantive = len(user_text.strip()) >= 2 and user_text_norm not in {"no", "not yet", "later"}
                if user_text_norm in {"yes", "y", "ready", "begin", "start", "go", "yeah", "yup"} or substantive:
                    # Start earliest item now
                    q = sr.get("queue", []) or []
                    if q:
                        item = q.pop(0)
                        sr["queue"] = q
                        prompt = begin_session(state, item)
                        if set_stage(state, "session_active"):
                            sp2 = SystemMessage(content=_build_system_prompt_session())
                            prior_messages.append(sp2)
                            _persist_chat_message(session_id, "system", sp2)
                        human_content = [{"type": "text", "text": f"Ask exactly this to begin: {prompt}"}]
                    else:
                        human_content = [{"type": "text", "text": "We don't have a clip ready yet. Give me a moment."}]
                else:
                    human_content = [{"type": "text", "text": "Please confirm you're ready and we'll begin: say 'ready' or 'begin'."}]
            else:
                # Non-kickoff idle path: due check or small talk
                due = get_next_due(state, now=now)
                if due is not None:
                    item = pop_next_due(state, now=now)
                    if item is not None:
                        prompt = begin_session(state, item)
                        if set_stage(state, "session_active"):
                            prior_messages.append(SystemMessage(content=_build_system_prompt_session()))
                        human_content = [{"type": "text", "text": f"Ask exactly this to begin the next session: {prompt}"}]
                    else:
                        human_content = [{"type": "text", "text": "Acknowledge and offer a brief supportive remark while waiting."}]
                else:
                    if set_stage(state, "waiting"):
                        spw = SystemMessage(content=_build_system_prompt_waiting())
                        prior_messages.append(spw)
                        _persist_chat_message(session_id, "system", spw)
                    if user_text_norm in {"begin", "start", "next", "go", "ready"}:
                        msg = "Great—I'm ready when you are. We'll start as soon as the next memory check is scheduled."
                    else:
                        msg = "We're giving your mind a short breather. When you're ready, say 'next' and we'll continue."
                    human_content = [{"type": "text", "text": msg}]

        # Append control instruction only; do not re-add system prompts on chat turns
        next_hm = HumanMessage(content=human_content)
        state["messages"] = prior_messages + [next_hm]  # type: ignore[index]
        _persist_chat_message(session_id, "user", next_hm)
        # Persist updated SR slice for this turn
        _append_and_persist_sr_state(session_id, state)

    async for part in parse_langgraph_stream(
        graph.astream(state, stream_mode="messages")
    ):
        yield part


async def main():
    """CLI for SR Agent runner (kickoff/chat/interactive)."""
    import argparse

    parser = argparse.ArgumentParser(description="SR Agent (kickoff once, then chat)")
    parser.add_argument("--session-id", help="Session ID for conversation persistence")
    parser.add_argument("--chat", help="Send a single chat message (assumes kickoff already run)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Interactive mode: runs kickoff, then loops over chat inputs",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive mode",
    )
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    fast = os.getenv("SR_FAST_START", "0").strip().lower() in {"1", "true", "yes"}
    mode = "FAST-START" if fast else "READINESS-GATED"
    print(f"Kickoff mode: {mode}")

    if args.chat:
        async for part in chat(session_id, args.chat):
            print("Coach:", part)
    elif args.interactive:
        print("Starting SR session...")
        print("- Kickoff will "+("start immediately with the first question." if fast else "ask if you're ready first."))
        print("- Type your responses. To start after readiness prompt, type 'ready'. Ctrl+C to quit.")
        try:
            async for part in kickoff(session_id):
                print("Coach:", part)
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                async for part in chat(session_id, user_input):
                    print("Coach:", part)
        except KeyboardInterrupt:
            print("\nSession ended. Your conversation is saved.")


if __name__ == "__main__":
    asyncio.run(main())


