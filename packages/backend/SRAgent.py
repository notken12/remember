#!/usr/bin/env python3

"""
Spaced Retrieval (SR) Agent skeleton.

This module defines the public interface and internal structure for a Spaced
Retrieval therapy agent that runs after an Episodic Specificity Induction (ESI)
session has completed. Importantly, SR does not consume or depend on any ESI
outputs. SR independently sources clips from Supabase (for now via a
hardcoded/known table and columns; in the future, via dynamic discovery of
recent/past clips).

State and I/O model:
    - This agent is intended to run under LangGraph with an AgentGraphState
      (or equivalent) provided by the orchestrator.
    - During a kickoff or chat call, the runtime will LOAD the graph state from
      Supabase at the beginning and SAVE it back at the end. Within the turn,
      SRAgent methods should only READ and UPDATE the in-memory graph state; do
      not perform direct persistence.

The SR agent schedules short, repeated recall sessions for selected clips using
increasing time intervals (e.g., 30s → 1m → 2m → 4m), adapting progression
based on patient performance.

NOTE: This file intentionally provides method signatures and detailed
docstrings only. No method contains business logic yet. A separate
implementation pass will fill these in and wire up streaming/tool-calls and
API endpoints as needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING
import heapq
from supabase import create_client, Client

# External dependencies used elsewhere in the codebase
from dotenv import load_dotenv

# Local types/modules (importable because ESIAgent adjusts sys.path to include chat/)
from VideoClip import VideoClip
from SRQuestionGenerator import QuestionGenerator
from Question import Question

if TYPE_CHECKING:
    from chat.ChatSession import ChatSession
    from chat.ChatMessage import ChatMessage


load_dotenv()


# ---------------------------------------------------------------------
# Phase 0: AgentGraphState schema and defaults (LangGraph scaffolding)
# ---------------------------------------------------------------------

# Typed, JSON-serializable shapes for the state managed by LangGraph.
try:
    # Python 3.8+ has TypedDict in typing; prefer it for shape declarations
    from typing import TypedDict
except Exception:  # pragma: no cover
    TypedDict = dict  # type: ignore


class QuestionState(TypedDict):
    """Serializable representation of a cue-based question.

    Keys:
        text_cue: The question text shown to the user.
        answer: A short, factual answer aligned with the clip content.
    """

    text_cue: str
    answer: str


class QueueItemState(TypedDict):
    """Serializable representation of a scheduled spaced-retrieval item.

    Keys:
        clip_id: UUID string referencing a video in Supabase.
        next_at: ISO-8601 timestamp indicating when the item becomes due.
        interval_index: Index into `sr.interval_seconds` for this schedule.
        attempt_count: Number of prior attempts during this SR run.
        questions: Ordered list of `QuestionState` for this clip.
        meta: Free-form metadata for HUD ranges, scoring traces, etc.
    """

    clip_id: str
    next_at: str
    interval_index: int
    attempt_count: int
    questions: List[QuestionState]
    meta: Dict[str, Any]


class ActiveSessionExchange(TypedDict, total=False):
    """One Q&A exchange inside an active clip session."""

    question: str
    user: str
    assessment: str


class ActiveSessionState(TypedDict):
    """Serializable representation of an ongoing clip Q&A session.

    Keys:
        clip_id: UUID of the clip being reviewed.
        q_index: Index of the next question to ask (0-based).
        exchanges: List of completed exchanges for audit and scoring.
    """

    clip_id: str
    q_index: int
    exchanges: List[ActiveSessionExchange]


class SRState(TypedDict, total=False):
    """Top-level SR slice of the graph state."""

    queue: List[QueueItemState]
    interval_seconds: List[int]
    clip_questions: Dict[str, List[QuestionState]]
    selected_clips: List[str]
    active_session: Optional[ActiveSessionState]
    last_activity_at: str


class ChatState(TypedDict, total=False):
    """Chat/session slice of the graph state (minimal for SR)."""

    session_id: str


class AgentGraphState(TypedDict, total=False):
    """Complete agent graph state as seen by the LangGraph runtime."""

    sr: SRState
    chat: ChatState


# Default constants for SR configuration
DEFAULT_INTERVAL_SECONDS: List[int] = [30, 60, 120, 240]
DEFAULT_QUESTIONS_PER_CLIP: int = 4
DEFAULT_CANDIDATE_LIMIT: int = 20


@dataclass
class QueueItem:
    """A single scheduled retrieval entry in the priority queue.

    Fields capture everything needed to execute one recall mini-session for a
    given clip, and to reschedule it based on performance.

    Attributes:
        clip_id: UUID of the video clip in Supabase.
        questions: The ordered list of `Question` objects to use for this clip.
        next_at: The absolute time when this item becomes due for processing.
        interval_index: Index into `SRAgent.interval_seconds` indicating the
            current spaced-retrieval interval that produced `next_at`.
        attempt_count: How many recall attempts have been completed for this
            clip within the current session scope (for analytics/backoff).
        meta: Optional dict for implementation-specific metadata (e.g., HUD
            display ranges, model selection rationale, scoring traces).
    """

    clip_id: str
    questions: List[Question]
    next_at: datetime
    interval_index: int = 0
    attempt_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> Dict[str, Any]:
        """Serialize into a JSON-safe dict for persistence.

        Returns:
            A dictionary that can be embedded into a Supabase JSON column or a
            message payload. The `questions` will be stored minimally by their
            cue/answer; the `clip_id` is persisted by value.
        """
        return {
            "clip_id": self.clip_id,
            "next_at": self.next_at.isoformat(),
            "interval_index": int(self.interval_index),
            "attempt_count": int(self.attempt_count),
            "questions": [
                {"text_cue": q.text_cue, "answer": q.answer} for q in self.questions
            ],
            "meta": dict(self.meta) if self.meta is not None else {},
        }

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "QueueItem":
        """Reconstruct a `QueueItem` from JSON state.

        Args:
            state: Dictionary produced by `to_state`.

        Returns:
            A `QueueItem` instance ready to be used by the agent. The
            `questions` will be reconstructed into `Question` objects. Any
            unknown fields should be ignored.
        """
        # Extract and normalize fields with basic validation
        clip_id = str(state.get("clip_id", ""))
        next_at_str = str(state.get("next_at", ""))
        interval_index = int(state.get("interval_index", 0))
        attempt_count = int(state.get("attempt_count", 0))
        meta = state.get("meta", {}) or {}

        try:
            next_at_dt = datetime.fromisoformat(next_at_str.replace("Z", "+00:00"))
        except Exception:
            # Fallback to immediate due if timestamp missing/invalid
            next_at_dt = datetime.utcnow()
        # Normalize to naive UTC for consistent comparisons
        if getattr(next_at_dt, "tzinfo", None) is not None:
            try:
                next_at_dt = next_at_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                next_at_dt = datetime.utcnow()

        questions_payload = state.get("questions", []) or []
        questions: List[Question] = []
        for item in questions_payload:
            try:
                text_cue = str(item.get("text_cue", "")).strip()
                answer = str(item.get("answer", "")).strip()
            except Exception:
                text_cue = ""
                answer = ""
            questions.append(Question(video_clip=None, text_cue=text_cue, answer=answer))

        return QueueItem(
            clip_id=clip_id,
            questions=questions,
            next_at=next_at_dt,
            interval_index=interval_index,
            attempt_count=attempt_count,
            meta=meta,
        )


class SRAgent:
    """Spaced Retrieval therapy agent interface and state container.

    Responsibilities:
        - Initialize from repository data (Supabase) after a prior ESI step has
          completed. The SR agent does NOT receive data directly from the ESI
          agent and is NOT called by the ESI agent. Instead, a client/user
          triggers SR kickoff after ESI finishes, and SRAgent loads any needed
          inputs (e.g., selected clip UUIDs and metadata) from Supabase.
        - Prepare or fetch cue-based recall questions per clip via
          `SRQuestionGenerator.QuestionGenerator`.
        - Maintain a priority queue of recall tasks, scheduled per spaced
          retrieval intervals. When a task is due, conduct an interactive
          Q&A mini-session for that clip.
        - After each mini-session, evaluate performance, advance/regress the
          interval, and reschedule the clip accordingly.
        - Integrate with LangGraph: read and write state via an AgentGraphState
          structure during a single turn; rely on the LangGraph runtime to load
          the state at the beginning of the call and persist it at the end.

    Important integration notes (non-goals of this file):
        - Streaming/tool-call mechanics (e.g., HUD display, LangChain events)
          are intentionally not implemented here. This class should expose
          stable method signatures and return values that higher-level layers
          can use to drive streaming UIs and endpoints.
    """

    def __init__(
        self,
        *,
        session: Optional["ChatSession"] = None,
        session_id: Optional[str] = None,
        interval_seconds: Optional[List[int]] = None,
        questions_per_clip: int = 4,
    ) -> None:
        """Construct a new SR agent bound to a chat session and schedule policy.

        Args:
            session: Optional pre-initialized `ChatSession`. If provided, this
                instance will be used for persistence of the conversation.
            session_id: If `session` is not provided, a new `ChatSession` should
                be created for this UUID; otherwise a fresh session can be
                generated. The underlying `ChatSession` should upsert itself
                when first used.
            interval_seconds: Custom spaced retrieval schedule as a list of
                seconds for successive reviews (default: [30, 60, 120, 240]).
            questions_per_clip: Target number of cue-based questions to prepare
                per clip via the `QuestionGenerator`.

        Side effects (when implemented):
            - Initialize Supabase client via `ChatSession`.
            - Load any serialized SR state previously saved for this session and
              hydrate in-memory queues so chat resumes seamlessly.
        """
        # In-memory state (mirrors AgentGraphState.sr)
        self._interval_seconds: List[int] = (
            list(interval_seconds) if interval_seconds else list(DEFAULT_INTERVAL_SECONDS)
        )
        self._questions_per_clip: int = int(questions_per_clip)
        # Priority queue implemented as a binary heap of (next_at, seq, item)
        self._heap: List[Tuple[datetime, int, QueueItem]] = []
        self._heap_seq: int = 0  # tie-breaker to maintain stable ordering
        # Cache of prepared questions per clip_id
        self._clip_questions: Dict[str, List[Question]] = {}
        # Deterministic subset chosen for this SR run
        self._selected_clips: List[str] = []
        # Active session state (None or dict-like mirror of ActiveSessionState)
        self._active_session: Optional[Dict[str, Any]] = None
        # Timestamps and misc
        self._last_activity_at: Optional[datetime] = None
        # Chat session placeholder (LangGraph may manage persistence separately)
        self.session = session  # type: ignore[assignment]
        self.session_id = session_id
        # Clip source configuration (can be overridden by environment)
        self._video_table: str = os.getenv("SR_VIDEO_TABLE", "test_videos")
        self._video_ts_columns: List[str] = [
            os.getenv("SR_VIDEO_TS_COLUMN", "time_created"),
            "created_at",
            "inserted_at",
        ]

    # ---------------------------------------------------------------------
    # Clip discovery (Supabase) and deterministic selection
    # ---------------------------------------------------------------------

    def _load_candidate_clips_from_supabase(self, *, limit: int = DEFAULT_CANDIDATE_LIMIT) -> List[str]:
        """Load recent candidate clip IDs from Supabase for SR.

        The method attempts to order by a preferred timestamp column. If that
        fails, it falls back to a simple `select('id')` without ordering. Any
        errors or missing configuration result in an empty list (callers should
        handle fallback behavior).

        Args:
            limit: Maximum number of candidate IDs to fetch.

        Returns:
            List of clip UUID strings, most-recent-first when possible.
        """
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return []

        try:
            client: Client = create_client(url, key)
        except Exception:
            return []

        # Try ordering by preferred timestamp columns first
        for ts_col in self._video_ts_columns:
            try:
                query = client.table(self._video_table).select("id, {}".format(ts_col))
                query = query.order(ts_col, desc=True)
                if limit:
                    query = query.limit(limit)
                resp = query.execute()
                data = resp.data or []
                ids = [str(row["id"]) for row in data if row.get("id")]
                if ids:
                    return ids
            except Exception:
                continue

        # Final fallback: no ordering, only IDs
        try:
            query = client.table(self._video_table).select("id")
            if limit:
                query = query.limit(limit)
            resp = query.execute()
            data = resp.data or []
            ids = [str(row.get("id")) for row in data if row.get("id")]
            return ids
        except Exception:
            return []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def kickoff(
        self,
        *,
        hardcoded_subset_size: int = 3,
    ) -> str:
        """Initialize an SR run and return deterministic starting text.

        Invocation contract:
            - Called by the client/user AFTER an ESI pipeline has completed.
            - The ESI agent does not call this directly and no ESI outputs are
              passed as function parameters. Instead, this method must LOAD
              any required inputs from Supabase (e.g., selected clip IDs,
              annotations, or references).

        It should perform all necessary setup work and return a fixed,
        deterministic string suitable for the first assistant message. The
        string should not depend on model randomness.

        Behavior (to implement):
            1) Ensure the chat session record exists (idempotent) if required by
               the environment. Under LangGraph, session existence may be
               guaranteed by the runtime.
            2) Query Supabase to retrieve the authoritative inputs (e.g., list
               of selected clip UUIDs and any clip metadata required for SR). If
               no such data is found, fall back to a small, hardcoded subset for
               development/testing using `hardcoded_subset_size`.
            3) For the chosen clips, pre-generate cue-based questions using
               `QuestionGenerator` (respecting `questions_per_clip`).
            4) Enqueue each clip for its first review at interval index 0.
            5) Write all SR initialization data into the AgentGraphState so that
               the LangGraph runtime can persist it at the end of the call.

        Args:
            hardcoded_subset_size: Fallback number of clips to select if the
                expected Supabase inputs are missing or incomplete.

        Returns:
            A deterministic, human-friendly kickoff string such as:
            "We will practice recall for N short clips. When you're ready, say 'Begin'."

        Notes:
            - Do not perform any model inference here, to preserve determinism.
            - This method should update only the in-memory AgentGraphState;
              the LangGraph runtime will handle loading/saving to Supabase at
              turn boundaries.
            - Streaming of this response, if needed, is the responsibility of
              higher layers. This method returns the full text.
        """
        # Ensure any required session scaffolding exists (may be a no-op)
        try:
            self.ensure_session_saved()
        except Exception:
            # Non-fatal; proceed with best-effort kickoff
            pass

        # Load existing SR state if present (LangGraph usually handles this)
        try:
            self.load_sr_state()
        except Exception:
            # Non-fatal; continue with fresh in-memory state
            pass

        # Discover candidate clips from Supabase
        candidate_ids: List[str] = []
        try:
            candidate_ids = self._load_candidate_clips_from_supabase(limit=DEFAULT_CANDIDATE_LIMIT)
        except Exception:
            candidate_ids = []

        # Fallback to env-provided list if Supabase returns none
        if not candidate_ids:
            fallback = os.getenv("SR_FALLBACK_CLIP_IDS", "")
            if fallback:
                # Comma-separated UUIDs
                candidate_ids = [cid.strip() for cid in fallback.split(",") if cid.strip()]

        # Deterministic selection of subset
        try:
            selected = self.select_clip_subset(candidate_ids, max_clips=int(hardcoded_subset_size))
        except Exception:
            selected = []

        self._selected_clips = list(selected)

        # Prepare questions per selected clip (cached when possible)
        questions_map: Dict[str, List[Question]] = {}
        if selected:
            try:
                questions_map = self.prepare_questions_for_clips(selected, questions_per_clip=self._questions_per_clip)
            except Exception:
                questions_map = {cid: [] for cid in selected}

        # Enqueue each selected clip at interval index 0
        now_ref = datetime.utcnow()
        for cid in selected:
            qlist = questions_map.get(cid, []) or []
            try:
                self.enqueue_clip_for_spaced_retrieval(
                    cid,
                    qlist,
                    interval_index=0,
                    base_time=now_ref,
                    meta={},
                )
            except Exception:
                # Skip enqueue failures to avoid blocking all clips
                continue

        # Update last activity timestamp and project into graph state
        try:
            self._last_activity_at = datetime.utcnow()
            self.save_sr_state()
        except Exception:
            pass

        n = len(selected)
        return (
            f"We will practice recall for {n} short clip{'s' if n != 1 else ''}. "
            "When you're ready, say 'Begin'."
        )

    def chat(self, user_text: str) -> str:
        """Handle one conversational turn of the SR loop and return assistant text.

        Core loop outline (to implement):
            (a) Peek at the top of the priority queue.
                - If `now >= next_at` for the head item, pop it and process the
                  recall mini-session for that clip (go to (b)).
                - Otherwise, generate a brief small-talk response (go to (e)).
            (b) Optionally instruct the HUD to display a still or short segment
                from the clip (tool-call placeholder).
            (c) Conduct an interactive question session with the human using the
                prepared questions for that clip. This sub-flow may span
                multiple user turns in practice. For this skeleton, return a
                concise, non-streamed summary of what the model would say.
            (d) Evaluate performance and compute the next interval index:
                - If successful, advance to the next interval.
                - Otherwise, step back to the previous interval (with floor at 0).
                Enqueue the clip at the computed `next_at` time.
            (e) Update the AgentGraphState with any new messages and scheduling
                changes. The LangGraph runtime will persist the updated state at
                the end of the call.

        Args:
            user_text: The user's latest utterance.

        Returns:
            A single assistant-message text for this turn (non-streamed). The
            surrounding API layer may convert this into a streaming response and
            tool-calls when implementing the final product.
        """
        # No implementation yet
        pass

    # ---------------------------------------------------------------------
    # Clip selection and question preparation
    # ---------------------------------------------------------------------

    def select_clip_subset(self, all_clip_ids: List[str], max_clips: int) -> List[str]:
        """Choose which clips to include in the SR run.

        This method exists to encapsulate selection logic. For early versions,
        it can return the first `max_clips` or a fixed hardcoded subset. Later
        iterations could apply heuristics (e.g., diversity by annotation) or a
        model-based policy.

        Args:
            all_clip_ids: Candidate clip UUIDs.
            max_clips: Maximum number of clips to select.

        Returns:
            Ordered list of chosen clip UUIDs, length in [0, max_clips].
        """
        if max_clips <= 0:
            return []
        if not all_clip_ids:
            return []
        # Deterministic: take the first N in given order
        return list(all_clip_ids[:max_clips])

    def prepare_questions_for_clips(
        self,
        clip_ids: List[str],
        *,
        questions_per_clip: Optional[int] = None,
    ) -> Dict[str, List[Question]]:
        """Generate or fetch cue-based questions for each clip.

        For each `clip_id`, this method should use `QuestionGenerator` to
        produce a small set of specific, answerable questions suitable for
        retrieval practice. Implementations may cache results to avoid redundant
        generation across turns.

        Args:
            clip_ids: List of clip UUIDs to prepare questions for.
            questions_per_clip: Optional override for how many questions to
                prepare per clip; defaults to the agent's configuration.

        Returns:
            Mapping from `clip_id` → list of `Question` objects.
        """
        # Normalize and de-duplicate while preserving original order
        seen: set[str] = set()
        ordered_ids: List[str] = []
        for cid in clip_ids:
            if not cid:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            ordered_ids.append(cid)

        target_count = int(questions_per_clip) if questions_per_clip else self._questions_per_clip
        if target_count <= 0:
            target_count = self._questions_per_clip

        results: Dict[str, List[Question]] = {}

        for clip_id in ordered_ids:
            # Serve from cache when possible
            cached = self._clip_questions.get(clip_id)
            if isinstance(cached, list) and cached:
                results[clip_id] = list(cached[:target_count])
                continue

            try:
                clip = VideoClip(clip_id)
                generator = QuestionGenerator(clip)
                questions = generator.generate(num_questions=target_count)
                if questions:
                    # Cache full set; return up to target_count
                    self._clip_questions[clip_id] = list(questions)
                    results[clip_id] = list(questions[:target_count])
                else:
                    # Leave uncached to allow retry on subsequent calls
                    results[clip_id] = []
            except Exception:
                # Robust to per-clip failures; continue with next clip
                results[clip_id] = []

        return results

    # ---------------------------------------------------------------------
    # Scheduling and queue management
    # ---------------------------------------------------------------------

    def enqueue_clip_for_spaced_retrieval(
        self,
        clip_id: str,
        questions: List[Question],
        *,
        interval_index: int = 0,
        base_time: Optional[datetime] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a clip to the priority queue for spaced retrieval.

        Args:
            clip_id: UUID of the clip.
            questions: Prepared questions for this clip.
            interval_index: Which interval step to schedule next (0-based).
            base_time: Reference time for scheduling; defaults to "now".
            meta: Optional metadata (e.g., HUD time bounds) to attach to the queue item.

        Raises:
            ValueError: If `interval_index` is outside the supported schedule.

        Side effects (to implement):
            - Compute `next_at = base_time + interval_seconds[interval_index]`.
            - Push a `QueueItem` to the internal heap/ordered structure.
        """
        if interval_index < 0 or interval_index >= len(self._interval_seconds):
            raise ValueError("interval_index out of range for configured schedule")

        reference_time = base_time or datetime.utcnow()
        # Ensure naive UTC for comparisons
        if getattr(reference_time, "tzinfo", None) is not None:
            try:
                reference_time = reference_time.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                reference_time = datetime.utcnow()

        delay_seconds = int(self._interval_seconds[interval_index])
        next_at = reference_time + timedelta(seconds=delay_seconds)

        item = QueueItem(
            clip_id=clip_id,
            questions=list(questions),
            next_at=next_at,
            interval_index=interval_index,
            attempt_count=0,
            meta=dict(meta) if meta else {},
        )

        self._heap_seq += 1
        heapq.heappush(self._heap, (item.next_at, self._heap_seq, item))

    def get_next_due_item(self, *, now: Optional[datetime] = None) -> Optional[QueueItem]:
        """Return the next due queue item if any (peek without removal).

        Args:
            now: Optional override for current time.

        Returns:
            The head `QueueItem` if it is due (`now >= next_at`); otherwise
            `None`. This does not remove the item from the queue.
        """
        if not self._heap:
            return None
        reference_time = now or datetime.utcnow()
        if getattr(reference_time, "tzinfo", None) is not None:
            try:
                reference_time = reference_time.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                reference_time = datetime.utcnow()
        head_next_at, _seq, head_item = self._heap[0]
        return head_item if reference_time >= head_next_at else None

    def pop_next_due_item(self, *, now: Optional[datetime] = None) -> Optional[QueueItem]:
        """Pop and return the next due item if ready; else return None.

        Args:
            now: Optional override for current time.

        Returns:
            The removed `QueueItem` if due, otherwise `None`.
        """
        if not self._heap:
            return None
        reference_time = now or datetime.utcnow()
        if getattr(reference_time, "tzinfo", None) is not None:
            try:
                reference_time = reference_time.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                reference_time = datetime.utcnow()
        head_next_at, _seq, head_item = self._heap[0]
        if reference_time >= head_next_at:
            _ = heapq.heappop(self._heap)
            return head_item
        return None

    # ---------------------------------------------------------------------
    # Single-clip session flow
    # ---------------------------------------------------------------------

    def conduct_clip_session(self, item: QueueItem) -> str:
        """Run the mini-session for a single clip and return assistant text.

        Steps (to implement):
            1) Optionally issue a tool-call to display a still or short segment
               from the clip on the user's HUD.
            2) Conduct an interactive Q&A loop using `item.questions`.
            3) Summarize the results into a concise assistant message for this
               turn. Persist transcript and scoring as needed.

        Args:
            item: The queue entry describing which clip and which questions.

        Returns:
            A concise assistant message summarizing the interaction for this
            turn. Streaming layers may yield intermediate tokens separately.
        """
        # No implementation yet
        pass

    def display_clip_to_hud(
        self,
        clip_id: str,
        *,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None,
    ) -> None:
        """Placeholder hook to display clip content on the HUD.

        This method does not implement actual streaming/tool-call logic. It is
        provided so higher layers can override/monkey-patch or intercept calls
        and forward them to the appropriate runtime (e.g., a LangChain tool,
        glasses SDK, or a UI event bus).

        Args:
            clip_id: UUID of the video to display.
            start_seconds: Optional start time for displaying a short segment.
            end_seconds: Optional end time for the segment.
        """
        # No implementation yet
        pass

    def ask_questions_interactively(self, questions: List[Question]) -> Dict[str, Any]:
        """Conduct a Q&A loop and return a structured transcript/result.

        This method encapsulates the turn-by-turn interaction needed to walk a
        patient through cue-based questions for a single clip. In production it
        will leverage a conversational model with streaming and tool-calls.

        Returns (to define when implemented):
            A dictionary such as:
            {
                "exchanges": [
                    {"question": str, "user": str, "assessment": str},
                    ...
                ],
                "score": float,  # Optional normalized score 0..1
                "success": bool, # Whether to advance the interval
            }
        """
        # No implementation yet
        pass

    def evaluate_performance(self, session_result: Dict[str, Any]) -> bool:
        """Determine whether the patient performed well enough to advance.

        Args:
            session_result: The dictionary produced by `ask_questions_interactively`.

        Returns:
            True if the interval should advance; False if it should regress or
            remain at the same level.

        Notes:
            - Early versions may use a simple heuristic (e.g., >= 70% correct or
              strong engagement). Later versions may perform semantic
              comparison or rubric-based scoring.
        """
        # No implementation yet
        pass

    def schedule_next_review(
        self,
        *,
        clip_id: str,
        current_interval_index: int,
        success: bool,
        questions: List[Question],
        now: Optional[datetime] = None,
    ) -> None:
        """Add the clip back to the queue at the appropriate next interval.

        Policy (to implement):
            - If `success` is True: advance to min(current + 1, last index).
            - If `success` is False: regress to max(current - 1, 0).
            - Compute new `next_at` using `interval_seconds[next_index]`.
            - Enqueue with updated `interval_index` and incremented `attempt_count`.
        """
        # No implementation yet
        pass

    # ---------------------------------------------------------------------
    # Waiting/small-talk behavior
    # ---------------------------------------------------------------------

    def small_talk_turn(self) -> str:
        """Generate a brief, supportive small-talk message while waiting.

        This method is used when no queue item is due yet. It should produce a
        single, self-contained assistant message that acknowledges the ongoing
        practice and optionally asks a gentle, low-effort question (e.g., about
        comfort, readiness, or environment).

        Returns:
            A short assistant message string suitable for direct display or
            streaming by higher layers.
        """
        # No implementation yet
        pass

    # ---------------------------------------------------------------------
    # Persistence helpers (Supabase-backed session and SR state)
    # ---------------------------------------------------------------------

    def ensure_session_saved(self) -> None:
        """Ensure session scaffolding exists (LangGraph environments may no-op).

        In a LangGraph + AgentGraphState setup, session creation/upsert and
        persistence are typically managed by the runtime at the edges of the
        call. Implementations may treat this as a no-op or as a lightweight
        validation that the expected state keys are present in the graph state.
        """
        # No implementation yet
        pass

    def load_sr_state(self) -> None:
        """Hydrate in-memory fields from AgentGraphState (LangGraph-managed).

        Under LangGraph, the graph state will already be loaded at the start of
        the call. This method should only read from that in-memory state and
        populate local structures (e.g., priority queue) as needed. It should
        not perform direct database I/O.

        Expected AgentGraphState keys used:
            - sr.interval_seconds: List[int]
            - sr.queue: List[QueueItemState]
            - sr.clip_questions: Dict[str, List[QuestionState]]
            - sr.selected_clips: List[str]
            - sr.active_session: ActiveSessionState | None
            - sr.last_activity_at: str (ISO) | missing

        Implementations should ensure safety if keys are missing.
        """
        # This method will be completed when the AgentGraphState instance is
        # passed into SRAgent. For now, leave logic to phase-2/3 integration.
        return

    def save_sr_state(self) -> None:
        """Project in-memory changes back into AgentGraphState (no direct I/O).

        In LangGraph, this method should only update the provided in-memory
        AgentGraphState with the latest queue contents, interval indices, and
        any clip/question caches. The orchestrator will persist state at the end
        of the call.

        Expected AgentGraphState keys to write:
            - sr.interval_seconds
            - sr.queue (serialize QueueItem → QueueItemState)
            - sr.clip_questions (serialize Question → QuestionState)
            - sr.selected_clips
            - sr.active_session
            - sr.last_activity_at (ISO timestamp)
        """
        # This method will be completed when we accept an AgentGraphState
        # instance in the public API and can write into it. Left for next phase.
        return

    def save_turn_messages(self, user_text: str, assistant_text: str) -> None:
        """Record turn messages into AgentGraphState; runtime persists as needed.

        In a LangGraph integration, higher layers may log message deltas and
        streaming tokens. This helper can consolidate the final user/assistant
        texts into the AgentGraphState so the runtime can persist them at the
        end of the call. No direct database writes should occur here.
        """
        # No implementation yet
        pass


def main() -> None:
    """Optional manual test harness entrypoint (no implementation).

    In a future implementation, this could:
        - Initialize `SRAgent` with a test session
        - Call `kickoff()` with a fixed set of clip IDs
        - Invoke `chat()` with sample inputs and print outputs
    """
    # No implementation yet
    pass


if __name__ == "__main__":
    main()


