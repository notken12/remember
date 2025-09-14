#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    AsyncGenerator,
    TypedDict,
)
import uuid
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from postgres import AsyncPostgresSaver
from parsing import parse_langgraph_stream
from protocol import StreamProtocolPart

# Supabase and video handling
from supabase import Client, create_client
from VideoClip import VideoClip

load_dotenv()

# Validate environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
database_url = os.getenv("DATABASE_URL")

if not all([supabase_url, supabase_key, database_url]):
    raise ValueError(
        "Required environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL"
    )

supabase: Client = create_client(supabase_url, supabase_key)


def _build_system_prompt() -> str:
    """Create the ESI therapist system prompt."""
    return (
        "You are an expert Episodic Specificity Induction (ESI) therapist helping a patient with early-stage Alzheimer's prepare for a Subsequent Retrieval (SR) session. "
        "Your aims: (1) gently cue vivid, specific memories; (2) scaffold sensory detail (sight, sound, touch, smell, taste), spatial/temporal anchors, and social/goal context; (3) cultivate safety and agency; (4) keep responses concise and momentum-building.\n\n"
        "Therapeutic style: warm, validating, non-judgmental, collaborative. Ask one clear question at a time. Encourage but never pressure. If distress surfaces, acknowledge it, downshift pace, and offer grounding (breath, present-moment sensory check).\n\n"
        "ESI priorities:\n"
        "- Sensory detail: colors, textures, sounds, temperature, smells, tastes\n"
        "- Specific where/when: location layout, time of day, season, sequence\n"
        "- Social/goal: who was there, what you/they wanted, interactions\n"
        "- Emotion and meaning: gentle curiosity; label emotions simply when invited\n"
        "- Safety: avoid overwhelming content; titrate and contain if needed\n\n"
        "Conversation rules:\n"
        "- Keep replies 1–3 short paragraphs or a short list.\n"
        "- Ask only one follow-up question.\n"
        "- Prefer simple, concrete language.\n"
        "- If memory is vague, offer options (e.g., sights, sounds, people) and let the patient choose.\n"
        "- If stuck, suggest a tiny step (notice lighting, a color, a voice).\n"
    )


class MediaPart(TypedDict):
    type: Literal["media"]
    mime_type: Literal["video/mp4"]
    data: str


def prepare_video_context(memory_uuids: List[str]) -> List[MediaPart]:
    media_parts = []

    for uuid_val in memory_uuids:
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
            print(f"Failed to prepare video {uuid_val}: {e}")
            continue

    return media_parts


def extract_memories(
    start_time_iso: Optional[str] = None,
    end_time_iso: Optional[str] = None,
    limit: Optional[int] = None,
    max_items: int = 3,
) -> List[MediaPart]:
    """Implementation of memory extraction."""
    print("Fetching annotated videos...")
    candidates = fetch_annotated_videos(start_time_iso, end_time_iso, limit)
    print("Selecting memories...")
    selected = select_memories_with_gemini(candidates, max_items)

    # Add annotations to selected memories
    uuid_to_annotation = {c["uuid"]: c.get("annotation", "") for c in candidates}
    for item in selected:
        if item["uuid"] in uuid_to_annotation:
            item["annotation"] = uuid_to_annotation[item["uuid"]]

    print("Preparing video context...")
    video_media_parts = prepare_video_context([item["uuid"] for item in selected])

    return video_media_parts


def fetch_annotated_videos(
    start_time_iso: Optional[str] = None,
    end_time_iso: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch annotated videos from Supabase."""
    query = (
        supabase.table("videos")
        .select("id, annotation, time_created")
        .not_.is_("annotation", "null")
        .neq("annotation", "")
    )

    if start_time_iso:
        query = query.gte("time_created", start_time_iso)
    if end_time_iso:
        query = query.lte("time_created", end_time_iso)
    if limit is not None:
        query = query.limit(limit)

    response = query.execute()
    data = response.data or []

    return [
        {
            "uuid": str(row["id"]),
            "annotation": str(row["annotation"]).strip(),
            "created_at": row.get("time_created"),
        }
        for row in data
        if row.get("id") and row.get("annotation")
    ]


def select_memories_with_gemini(
    candidates: Sequence[Dict[str, Any]], max_items: int
) -> List[Dict[str, str]]:
    """Use Gemini to select ESI-conducive memories."""
    if not candidates:
        return []

    dataset_json = json.dumps(
        [{"uuid": c["uuid"], "annotation": c["annotation"]} for c in candidates],
        ensure_ascii=False,
    )

    prompt = f"""
You are an expert clinician facilitating Episodic Specificity Induction (ESI) therapy.
You are given brief annotations of first-person video clips captured by smart glasses.

Your task: Choose up to {max_items} clips that are most conducive to ESI.

Prioritize clips whose annotations indicate:
- Rich, concrete sensory detail (visual, auditory, tactile, olfactory)
- Clear spatiotemporal specificity (where and when)
- Goal-directed or socially interactive moments that can be elaborated
- Emotionally salient yet safe content (avoid overwhelming distress or trauma)
- Distinctiveness/variability across clips to cover diverse contexts

Output format requirements (must follow exactly):
- Return ONLY a JSON array (no extra text, no markdown) where each item is an object:
{{"uuid": "<clip_uuid>", "reasoning": "1–2 sentences explaining why this clip is ideal for ESI"}}

Here are the candidate clips (JSON):
{dataset_json}
"""
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    response = model.invoke(prompt)
    text = response.content

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to salvage JSON
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            parsed = json.loads(text[start_idx : end_idx + 1])
        else:
            return []

    results = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        uuid_val = item.get("uuid") or item.get("id")
        reasoning_val = item.get("reasoning")
        if uuid_val and reasoning_val:
            results.append(
                {"uuid": str(uuid_val), "reasoning": str(reasoning_val).strip()}
            )

    return results[:max_items]


class ESIAgent:
    """
    Simplified ESI Agent using LangGraph ReAct agents for memory extraction and therapy.
    Uses AsyncPostgresSaver for automatic session management.
    """

    def __init__(self, session_id: Optional[str] = None):
        # Environment setup
        self.database_url = os.getenv("DATABASE_URL", "")

        if not all(
            [
                self.database_url,
            ]
        ):
            raise ValueError(
                "Required environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL"
            )

        # Initialize clients

        # Session configuration
        self.session_id = session_id or f"esi_session_{datetime.now().isoformat()}"
        self.config = {"configurable": {"thread_id": self.session_id}}

        # Memory context
        self.selected_memories: List[Dict[str, Any]] = []

        # Initialize checkpointer and agent once to avoid prepared statement conflicts
        self._checkpointer = None
        self._agent = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the checkpointer and agent once for the session."""
        if self._checkpointer is None:
            # Create a single connection that will be reused
            from psycopg import Connection
            from psycopg.rows import dict_row

            conn = Connection.connect(
                self.database_url,
                autocommit=True,
                prepare_threshold=0,  # Disable prepared statements
                row_factory=dict_row,
                connect_timeout=30,
                options="-c statement_timeout=300000",
            )
            self._checkpointer = AsyncPostgresSaver(conn, None)
            self._checkpointer._conn_string = self.database_url

            # Create the agent with the checkpointer
            self._agent = create_react_agent(
                model=init_chat_model(
                    model="gemini-2.5-flash",
                    model_provider="google_genai",
                ),
                tools=[],
                checkpointer=self._checkpointer,
            )

    def __del__(self):
        """Clean up the database connection when the agent is destroyed."""
        if self._checkpointer and hasattr(self._checkpointer, "conn"):
            try:
                self._checkpointer.conn.close()
            except:
                pass

    async def kickoff(self) -> AsyncGenerator[StreamProtocolPart, None]:
        """Start the therapy session with therapist speaking first."""
        system_prompt = _build_system_prompt()

        kickoff_message = "Finally, greet the patient warmly and ask ONE gentle, concrete recall question based on the prepared memories."

        print("Extracting memories...")
        video_media_parts = extract_memories()

        # Use the persistent agent instead of creating a new one
        async for part in parse_langgraph_stream(
            self._agent.astream(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=[
                                {"type": "text", "text": kickoff_message},
                                *video_media_parts,
                            ]
                        ),
                    ],
                },
                config=self.config,
                stream_mode="messages",
            )
        ):
            yield part

    async def chat(self, user_message: str) -> AsyncGenerator[StreamProtocolPart, None]:
        """Continue the therapy conversation."""
        # Include video context if available
        message = HumanMessage(content=user_message)

        # Use the persistent agent instead of creating a new one
        snapshot = self._agent.get_state(self.config)
        graph_state = snapshot.values
        graph_state["messages"] = graph_state["messages"] + [message]

        async for part in parse_langgraph_stream(
            self._agent.astream(graph_state, config=self.config, stream_mode="messages")
        ):
            yield part


async def main():
    """CLI interface for the ESI Agent."""
    import argparse

    parser = argparse.ArgumentParser(description="ESI Agent with LangGraph")
    parser.add_argument("--session-id", help="Session ID for conversation persistence")
    parser.add_argument(
        "--interactive", action="store_true", default=True, help="Interactive chat mode"
    )
    parser.add_argument("--chat", help="Single message to send")

    args = parser.parse_args()

    session_id = args.session_id or uuid.uuid4()
    agent = ESIAgent(session_id=session_id)
    print(f"Session ID: {session_id}")

    if args.chat:
        async for part in agent.chat(args.chat):
            print("Therapist:", part)
    elif args.interactive:
        print("Starting ESI therapy session...")
        try:
            # Therapist initiates
            async for part in agent.kickoff():
                print("Therapist:", part)

            # Interactive loop
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                async for part in agent.chat(user_input):
                    print("Therapist:", part)

        except KeyboardInterrupt:
            print("\nSession ended. Your conversation is saved.")


if __name__ == "__main__":
    asyncio.run(main())
