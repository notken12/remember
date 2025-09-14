"""
AssistantAgent: A LangGraph-based assistant for day-to-day support.

This module defines the scaffolding for an assistant agent designed to help
people with Alzheimer's in their daily lives. The agent mirrors the
architecture used by `esi_agent.py`:

- Uses a LangGraph with an agent node (LLM) and a tools node (ToolNode)
- Streams responses using `parse_langgraph_stream` to yield `StreamProtocolPart`
- Persists conversation state to Supabase (`chat_sessions`, `chat_messages`)

This file contains only function and class scaffolding with comprehensive
docstrings. Implementations are intentionally omitted for clarity and phased
development.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict, TYPE_CHECKING
import os
import uuid

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

from supabase_client import supabase
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from parsing import parse_langgraph_stream
# For type checking only to avoid runtime import cycles
if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocol import StreamProtocolPart
from agent_state import State, get_state_from_supabase
from chat.ChatSession import ChatSession
from chat.ChatMessage import ChatMessage


class Video(TypedDict):
    """Typed shape for a video record used by the assistant.

    Keys:
    - uuid: The unique identifier for the video (string UUID from database)
    - annotation: The human-readable annotation text describing the clip
    - created_at: ISO timestamp or database timestamp string for when the video was created
    """

    uuid: str
    annotation: str
    created_at: Optional[str]


# Tooling: minimal tool for directing the client to display a given video UUID
@tool
def select_video(video_uuid: str) -> Dict[str, str]:
    """Request the client to display a specific video by UUID.

    Parameters
    ----------
    video_uuid : str
        The UUID of the video that should be displayed by the client UI.

    Returns
    -------
    Dict[str, str]
        Structured directive: {"uuid": <video_uuid>} for downstream handling.
    """
    return {"uuid": str(video_uuid)}


class AssistantAgent:
    """A gentle daily-life assistant for people with Alzheimer's.

    The agent is designed to:
    - Answer everyday questions in a supportive, concise, and concrete manner
    - Optionally request a specific video (by UUID) to be shown by the client
    - Otherwise provide short textual guidance referencing the best-matching clip

    Architectural notes
    -------------------
    - The agent mirrors `esi_agent.py` by using a LangGraph with an agent node
      and a tools node, routed via `tools_condition`.
    - The conversation state is persisted to Supabase via `chat_messages` and
      `chat_sessions` tables, enabling continuity across requests.
    - Responses are streamed to the caller using `parse_langgraph_stream`,
      yielding `StreamProtocolPart` elements.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_corpus_items: int = 200,
        max_display_items: int = 3,
    ) -> None:
        """Initialize the AssistantAgent and prepare its graph.

        Parameters
        ----------
        session_id : Optional[str]
            An optional session ID for conversation persistence. If not
            provided, the eventual implementation may generate one as needed.
        model : str
            The model identifier to use with the underlying chat model
            (e.g., "gemini-2.5-flash").
        max_corpus_items : int
            A soft upper bound for how many annotated videos to include as
            inline context to the model during a query to manage token usage.
        max_display_items : int
            A limit for how many top candidate clips to highlight or consider
            prominently per query.

        Notes
        -----
        - The implementation should load necessary environment variables,
          initialize any required clients (e.g., Supabase), and compile a
          LangGraph instance that binds the tools.
        - No external side effects should occur beyond preparing in-memory
          structures; database writes should happen during `query()`.
        """
        # Store configuration for later phases
        self.session_id = session_id
        self.model = model
        self.max_corpus_items = max_corpus_items
        self.max_display_items = max_display_items

        # Shared clients and placeholders
        self.supabase = supabase
        self._graph = None

        # Compile a minimal graph mirroring esi_agent shape (agent <-> tools)
        self._build_graph()

    def _build_system_prompt(self) -> str:
        """Create the system prompt instructing the assistant's behavior.

        The prompt should:
        - Adopt a warm, supportive tone appropriate for assisting someone with
          Alzheimer's, emphasizing clarity, kindness, and agency.
        - Instruct the model to keep replies brief (1–3 short paragraphs or a
          short list) and ask at most one follow-up question when useful.
        - Explain tool/output policies:
          - If a specific clip likely answers the question, call the tool
            `select_video` exactly once with the best `video_uuid` so the client
            can display it.
          - Always provide a brief textual context that explains what/where the
            clip shows (e.g., "wallet on the dresser"), even when you call the
            tool.
          - If uncertain, provide a concise textual response grounded in the
            provided annotations, optionally referencing relevant UUIDs, and ask
            a single clarifying question.
        - Encourage concrete sensory and spatial context when describing scenes.

        Returns
        -------
        str
            The system prompt content to prepend before user messages.
        """
        # Therapist-style, gentle daily assistant prompt tailored for Alzheimer's support
        return (
            "You are a gentle, supportive daily assistant helping a person living with early-stage Alzheimer's. "
            "Your goals: (1) answer practical questions simply and kindly; (2) reduce confusion with concrete, brief guidance; "
            "(3) when appropriate, request the app to show a specific recent video by calling the select_video tool; "
            "(4) keep momentum while honoring the person's agency and safety.\n\n"
            "Style: warm, validating, non-judgmental, collaborative. Prefer simple, concrete language. "
            "Use 1–3 short paragraphs or a short list. Ask at most one gentle clarifying question if needed.\n\n"
            "Video guidance: You will be given annotated video entries from smart glasses, each with a UUID. "
            "If a single clip clearly answers the question (e.g., where an item is), call select_video with that clip's UUID, and also provide a one-sentence description of what/where it shows. "
            "Otherwise, give a short textual answer grounded in the annotations (mention the best UUIDs) and optionally ask one clarifying question. "
            "Only reference UUIDs that are present in the provided context.\n\n"
            "Safety: If the user seems distressed or uncertain, slow down, acknowledge feelings briefly, and offer a small grounding step (e.g., notice a color or sound)."
        )

    def _fetch_annotated_videos(
        self,
        start_time_iso: Optional[str] = None,
        end_time_iso: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Video]:
        """Load annotated videos from the database for model context.

        This method mirrors `fetch_annotated_videos` in `esi_agent.py` to
        retrieve only those videos that have a non-empty `annotation` field.

        Parameters
        ----------
        start_time_iso : Optional[str]
            Lower bound for filtering videos by creation time (inclusive),
            expressed in ISO 8601 format.
        end_time_iso : Optional[str]
            Upper bound for filtering videos by creation time (inclusive),
            expressed in ISO 8601 format.
        limit : Optional[int]
            Maximum number of rows to fetch from the database.

        Returns
        -------
        List[Video]
            A list of video dictionaries with keys: `uuid`, `annotation`,
            `created_at`.

        Notes
        -----
        - The implementation should rely on Supabase queries to the `videos`
          table, align the output structure with `esi_agent.py`, and ensure
          empty or null annotations are excluded.
        - The results are intended to be embedded into the prompt context for
          each `query()` call.
        """
        query = (
            self.supabase.table("videos")
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
        rows = response.data or []

        results: List[Video] = []
        for row in rows:
            if not row.get("id") or not row.get("annotation"):
                continue
            results.append(
                Video(
                    uuid=str(row["id"]),
                    annotation=str(row["annotation"]).strip(),
                    created_at=row.get("time_created"),
                )
            )

        return results

    def _select_candidates_with_gemini(
        self, user_query: str, candidates: Sequence[Dict[str, Any]], max_items: int
    ) -> List[Dict[str, str]]:
        """Optionally shortlist top video candidates for a given query.

        This helper can reuse the approach in `esi_agent.py::select_memories_with_gemini`
        to choose up to `max_items` clips that are most likely to answer a user
        question. This can reduce token usage and improve determinism by
        presenting the LLM with a smaller set of high-signal options.

        Parameters
        ----------
        candidates : Sequence[Dict[str, Any]]
            A collection of candidate video records, typically from
            `_fetch_annotated_videos`, each containing `uuid` and `annotation`.
        max_items : int
            The maximum number of shortlisted candidates to return.

        Returns
        -------
        List[Dict[str, str]]
            A list of objects with keys:
            - `uuid`: The candidate clip UUID
            - `reasoning`: Brief explanation for why the clip is relevant

        Notes
        -----
        - Implementations should handle malformed model output gracefully and
          return an empty list when selection fails.
        - If not used, the assistant may rely on the full corpus within
          `_build_system_prompt` and/or user messages.
        """
        if not candidates:
            return []

        import json as _json

        dataset_json = _json.dumps(
            [{"uuid": c.get("uuid"), "annotation": c.get("annotation")} for c in candidates],
            ensure_ascii=False,
        )

        prompt = f"""
You are a helpful assistant selecting up to {max_items} video clips that best answer the user's practical question.
User question: {user_query}

You are given brief annotations for first-person videos captured by smart glasses.

Choose clips that likely contain the sought item or context, prioritizing:
- Specific references to objects/locations, concrete sensory or spatial detail
- Recency and clarity if discernible from annotations
- Diversity across suggestions when uncertain

Output format (must be ONLY a JSON array of objects):
{{"uuid": "<clip_uuid>", "reasoning": "one short sentence"}}

Here are candidate clips (JSON):
{dataset_json}
"""

        llm = init_chat_model(model=self.model, model_provider="google_genai")
        response = llm.invoke(prompt)
        text = getattr(response, "content", "")

        try:
            parsed = _json.loads(text)
        except Exception:
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    parsed = _json.loads(text[start_idx : end_idx + 1])
                except Exception:
                    return []
            else:
                return []

        results: List[Dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            uuid_val = item.get("uuid") or item.get("id")
            reasoning_val = item.get("reasoning")
            if uuid_val and reasoning_val:
                results.append({"uuid": str(uuid_val), "reasoning": str(reasoning_val).strip()})

        return results[: max_items or 0] if max_items else results

    def _build_graph(self) -> None:
        """Construct and compile the LangGraph for the assistant.

        The graph should:
        - Define an `agent` node that invokes the configured LLM over the
          current `State["messages"]`, appends the assistant's reply to the
          state, and persists it to Supabase.
        - Optionally include a `tools` node using `ToolNode` with no bound tools,
          preserving the graph shape used by `esi_agent.py` for future extension.
        - Route from START → agent, then conditionally to `tools` via
          `tools_condition`, then back to `agent` to continue the loop.

        Notes
        -----
        - The compiled graph should be stored on the instance (e.g., `self.graph`).
        - Any required clients or savers for persistence should be prepared
          beforehand in `__init__`.
        """
        # Establish the graph nodes and edges consistent with esi_agent.py.
        # Minimal no-op nodes are used here; implementations will be added later.

        def agent_node(state: State) -> State:
            """Core assistant node: invoke LLM over accumulated messages and persist.

            Invokes the configured chat model with the current state's messages,
            appends the AI message to the state, and writes the AI message to
            Supabase `chat_messages` with role "assistant".
            """
            llm = init_chat_model(model=self.model, model_provider="google_genai")
            llm = llm.bind_tools([select_video])
            message = llm.invoke(state["messages"])  # type: ignore[index]
            state["messages"].append(message)

            # Persist assistant message via ChatMessage (store raw text too)
            assistant_text = ""
            content = getattr(message, "content", "")
            if isinstance(content, list):
                # Extract text blocks if present
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        assistant_text += item.get("text", "")
            elif isinstance(content, str):
                assistant_text = content

            ChatMessage(
                content=assistant_text,
                session_id=str(state["session_id"]),
                role="assistant",
                data=message.__dict__,
            ).save_to_supabase()
            return state

        def tools_node(state: State) -> State:
            """Execute pending tool calls and persist tool outputs.

            Uses LangGraph's ToolNode to run tool calls produced by the agent
            (e.g., `select_video`) and appends their ToolMessage results to the
            state and Supabase.
            """
            node = ToolNode(tools=[select_video])
            tool_output = node.invoke(state)
            # ToolNode may return a list of ToolMessage or a dict with {'messages': [...]} depending on version
            if isinstance(tool_output, dict) and "messages" in tool_output:
                tool_messages = tool_output.get("messages", [])
            else:
                tool_messages = tool_output
            iterable = tool_messages if isinstance(tool_messages, list) else [tool_messages]
            state["messages"].extend(iterable)

            # Persist tool messages via ChatMessage (defensive against str/dict types)
            for tool_msg in iterable:
                if hasattr(tool_msg, "content"):
                    tool_text = str(getattr(tool_msg, "content"))
                else:
                    tool_text = str(tool_msg)

                if hasattr(tool_msg, "__dict__"):
                    data_payload = tool_msg.__dict__  # type: ignore[attr-defined]
                elif isinstance(tool_msg, dict):
                    data_payload = tool_msg
                else:
                    data_payload = {"content": tool_text}

                ChatMessage(
                    content=tool_text,
                    session_id=str(state["session_id"]),
                    role="tool",
                    data=data_payload,
                ).save_to_supabase()
            return state

        graph_builder = StateGraph(State)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tools_node)
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        self._graph = graph_builder.compile()

    async def query(
        self, user_query: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[StreamProtocolPart, None]:
        """Stream a response to a user query, optionally calling a tool.

        This endpoint mirrors `esi_agent.py` streaming behavior and persistence.
        It should:
        1) Ensure a `chat_session` exists for the `session_id`.
        2) Build messages consisting of:
           - A `SystemMessage` from `_build_system_prompt()`
           - A corpus context message listing annotated videos (capped by
             `max_corpus_items`) so the model can ground its reasoning
           - The user's query as a `HumanMessage`
        3) Persist the user message to `chat_messages` and stream the results of
           `self.graph.astream(..., stream_mode="messages")` via
           `parse_langgraph_stream`.

        Parameters
        ----------
        user_query : str
            The user's natural-language question (e.g., "Where is my wallet?").
        session_id : Optional[str]
            An existing session identifier to continue a conversation. If not
            provided, the implementation may generate a new one and persist it.

        Yields
        ------
        StreamProtocolPart
            An async sequence of streaming parts consisting of text deltas and
            tool call/input/output events. Tool outputs should include the
            structured response from `show_video` when invoked.

        Notes
        -----
        - On subsequent calls with the same `session_id`, the implementation
          should load previous messages using `get_state_from_supabase(session_id)`
          and append the new `HumanMessage` before streaming.
        - If the model determines that a specific clip solves the request, it
          should call `show_video` exactly once with the best `video_uuid`.
          Otherwise, it should reply with concise text that references the most
          relevant annotations.
        - The function does not return a final string; it yields a stream of
          protocol parts to the caller in real time.
        """
        # Ensure a session id and session record
        print(f"🔍 User query: {user_query}")
        sid = session_id or self.session_id or str(uuid.uuid4())
        ChatSession(session_id=sid).save_to_supabase()

        # Build system prompt and corpus context
        system_prompt = self._build_system_prompt()
        annotated_videos = self._fetch_annotated_videos(limit=self.max_corpus_items)

        # Optional shortlist of top candidates using the question context
        try:
            shortlist = self._select_candidates_with_gemini(
                user_query, annotated_videos, self.max_display_items
            )
        except Exception:
            shortlist = []

        print(f"🔍 Shortlist: {shortlist}")

        # Construct a compact corpus context message (shortlist + corpus)
        lines: List[str] = []
        if shortlist:
            lines.append("Top candidate videos (by UUID):")
            for s in shortlist:
                lines.append(f"- {s.get('uuid')}: {s.get('reasoning', '')}")
            lines.append("")

        lines.append("Here are annotated videos from smart glasses to help with your answer:")
        for i, v in enumerate(annotated_videos):
            prefix = f"Video {i + 1} [{v['uuid']}]: "
            annotation = v.get("annotation", "")
            lines.append(prefix + annotation)
        corpus_context_text = "\n".join(lines)

        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=corpus_context_text),
        ]
        
        print(f"🔍 Initial messages: {initial_messages}")

        # Persist the user's query and add it to the state via ChatMessage
        user_message = HumanMessage(content=user_query)
        ChatMessage(
            content=user_query,
            session_id=sid,
            role="user",
            data=user_message.__dict__,
        ).save_to_supabase()

        # Load previous state if exists, then ALWAYS prepend system + corpus context
        try:
            graph_state = get_state_from_supabase(sid)
            graph_state["messages"] = initial_messages + graph_state["messages"] + [user_message]
        except Exception:
            graph_state = {
                "messages": initial_messages + [user_message],
                "session_id": sid,
            }  # type: ignore[typeddict-item]

        # Stream through the compiled graph and translate to protocol parts
        print(f"asdfasdfasdfasfadf Graph state messages: {graph_state['messages']}")
        async for part in parse_langgraph_stream(
            self._graph.astream(graph_state, stream_mode="messages")  # type: ignore[union-attr]
        ):
            yield part

def main():
    """Simple CLI for manual testing.

    Usage (from repository root or backend package):
        python -m packages.backend.AssistantAgent --chat "where is my wallet?"

    Options:
        --session-id <uuid>   Continue an existing session
        --chat "..."          Send a single user message and stream the reply
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="AssistantAgent CLI")
    parser.add_argument("--session-id", help="Session ID for persistence")
    parser.add_argument("--chat", help="Single message to send")
    args = parser.parse_args()

    agent = AssistantAgent(session_id=args.session_id)

    async def _run():
        if not args.chat:
            print("Please provide --chat \"your question\"")
            return
        async for part in agent.query(args.chat, session_id=args.session_id):
            print(part)

    asyncio.run(_run())


if __name__ == "__main__":
    main()