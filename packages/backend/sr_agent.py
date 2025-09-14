#!/usr/bin/env python3

import asyncio
import os
from typing import List, Literal, TypedDict, Dict, Any, Optional

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


