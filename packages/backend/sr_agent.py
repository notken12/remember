#!/usr/bin/env python3

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import List, Literal, TypedDict, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
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
    selected_clips: List[str]
    enqueued_clips: List[str]
    clip_questions: Dict[str, List[QuestionState]]
    queue: List[QueueItemState]
    active_session: Optional[ActiveSessionState]
    last_activity_at: str


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

    return sr


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

    # Video source configuration is read where discovery is performed; export defaults via env
    sr.setdefault("video_table", os.getenv("SR_VIDEO_TABLE", DEFAULT_VIDEO_TABLE))
    sr.setdefault("video_ts_column", os.getenv("SR_VIDEO_TS_COLUMN", DEFAULT_VIDEO_TS_COLUMN))


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
    except Exception:
        pass

    # Fallback ordering by created_at
    try:
        query = supabase.table(table).select("id, created_at").order("created_at", desc=True)
        if limit:
            query = query.limit(limit)
        resp = query.execute()
        data = resp.data or []
        ids = [str(row["id"]) for row in data if row.get("id")]
        return _dedupe_preserve_order(ids)
    except Exception:
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
    exchanges = list(sess.get("exchanges", []))
    exchanges.append({"question": question_text, "user": str(answer or "").strip(), "assessment": ""})
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
    """Return {score: float, success: bool} using 0.7 threshold on non-empty answers."""
    sr = ensure_sr_slice(state)
    sess = sr.get("active_session")
    if not isinstance(sess, dict):
        return {"score": 0.0, "success": False}
    qs = sess.get("questions", []) or []
    exchanges = list(sess.get("exchanges", []))
    total = max(1, len(qs))
    answered = 0
    for ex in exchanges:
        try:
            if str(ex.get("user", "")).strip():
                answered += 1
        except Exception:
            continue
    score = min(1.0, answered / float(total))
    return {"score": score, "success": score >= 0.7}


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


async def main():
    """Placeholder CLI for SR runner (will be expanded in subsequent phases)."""
    import argparse

    parser = argparse.ArgumentParser(description="SR Agent (scaffold)")
    parser.add_argument("--session-id", help="Session ID for conversation persistence")
    args = parser.parse_args()

    session_id = args.session_id or "sr_session"
    print(f"SR Agent scaffold ready. Session ID: {session_id}")


if __name__ == "__main__":
    asyncio.run(main())


