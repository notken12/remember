#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import google.generativeai as genai
from dotenv import load_dotenv
from supabase import Client, create_client
from VideoClip import VideoClip

# Make local "chat" package importable (for ChatSession/ChatMessage)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = os.path.join(CURRENT_DIR, "chat")
if CHAT_DIR not in sys.path:
    sys.path.append(CHAT_DIR)

from ChatSession import ChatSession  # type: ignore
from ChatMessage import ChatMessage  # type: ignore

# LangChain - Gemini chat model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


class ESIAgent:
    """
    Selects the most ESI-conducive memories from annotated smart-glasses videos.

    Responsibilities:
    - Fetch annotated videos from Supabase with optional time filtering
    - Craft an ESI-focused instruction to guide selection
    - Call Gemini 2.5 Flash to return clean JSON: [{"uuid", "reasoning"}, ...]
    """

    def __init__(
        self,
        *,
        session: Optional["ChatSession"] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.supabase_url: str = os.getenv("SUPABASE_URL", "")
        self.supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.model_name: str = "gemini-2.5-flash"
        self.table_name: str = "test_videos"
        # Default to created_at but permit override via CLI; keep legacy compat
        self.timestamp_column: str = "time_created"
        # Context mode: "video" | "annotation"
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("Supabase credentials are required (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_service_role_key)
        genai.configure(api_key=self.gemini_api_key)
        # Keep selected memories and prepared video handles for downstream use
        self.memories_context: List[Dict[str, Any]] = []
        self.video_files_context: List[Any] = []
        # LangChain media parts cache for selected videos
        self._langchain_media_parts: List[Dict[str, Any]] = []

        # Initialize chat session (persistent memory)
        self.session: Optional["ChatSession"] = session
        if self.session is None and session_id:
            if ChatSession is None:
                raise RuntimeError("ChatSession module not available for session initialization")
            self.session = ChatSession(session_id=session_id)

        else:
            self.session = ChatSession()

        # LangChain LLM setup (therapist chat)
        if ChatGoogleGenerativeAI is None:
            # Allow import-time absence; fail only when chat is invoked
            self.llm = None  # type: ignore
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.gemini_api_key,
            )

    def fetch_annotated_videos(
        self,
        *,
        start_time_iso: Optional[str] = None,
        end_time_iso: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch annotated videos from Supabase with optional ISO timestamp filtering.

        Returns a list of dicts with keys: id, annotation, created_at (if present).
        """
        query = (
            self.supabase
            .table(self.table_name)
            .select("id, annotation, {}".format(self.timestamp_column))
            .not_.is_("annotation", "null")
            .neq("annotation", "")
        )

        if start_time_iso:
            query = query.gte(self.timestamp_column, start_time_iso)
        if end_time_iso:
            query = query.lte(self.timestamp_column, end_time_iso)
        if limit is not None:
            query = query.limit(limit)

        print("🔎 Fetching annotated videos from Supabase...")
        response = query.execute()
        data = response.data or []
        try:
            print(f"✅ Retrieved {len(data)} candidate rows")
        except Exception:
            pass
        # Normalize keys and ensure types
        normalized: List[Dict[str, Any]] = []
        for row in data:
            if not row.get("id") or not row.get("annotation"):
                continue
            normalized.append(
                {
                    "uuid": str(row["id"]),
                    "annotation": str(row["annotation"]).strip(),
                    "created_at": row.get(self.timestamp_column),
                }
            )
        return normalized

    def _build_esi_instruction(self, candidates: Sequence[Dict[str, Any]], max_items: int) -> str:
        """
        Build the instruction for ESI selection with an explicit output contract.
        """
        dataset_json = json.dumps(
            [{"uuid": c["uuid"], "annotation": c["annotation"]} for c in candidates],
            ensure_ascii=False,
        )

        instruction = f"""
You are an expert clinician facilitating Episodic Specificity Induction (ESI) therapy.
You are given brief annotations of first-person video clips captured by smart glasses.

Your task: Choose up to {max_items} clips that are most conducive to ESI.

Prioritize clips whose annotations indicate:
- Rich, concrete sensory detail (visual, auditory, tactile, olfactory)
- Clear spatiotemporal specificity (where and when)
- Goal-directed or socially interactive moments that can be elaborated
- Emotionally salient yet safe content (avoid overwhelming distress or trauma)
- Distinctiveness/variability across clips to cover diverse contexts

De-emphasize:
- Vague, generic, or purely factual summaries with little imagery
- Repetitive scenes with minimal new detail across clips
- Content that appears unsafe or likely to trigger severe distress

Output format requirements (must follow exactly):
- Return ONLY a JSON array (no extra text, no markdown) where each item is an object:
  {{"uuid": "<clip_uuid>", "reasoning": "1–2 sentences explaining why this clip is ideal for ESI"}}
- Keep reasoning concise, focusing on ESI-relevant features (sensory detail, specificity, social/goal relevance, emotional salience, safety).

Here are the candidate clips (JSON):
{dataset_json}
"""
        return instruction

    def select_memories(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        max_items: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Call Gemini to select and return a clean JSON array of {"uuid", "reasoning"}.
        """
        if not candidates:
            return []

        prompt = self._build_esi_instruction(candidates, max_items=max_items)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "response_mime_type": "application/json",
            },
        )

        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or getattr(response, "candidates", None)
        if isinstance(text, list):
            # Fallback if SDK returns structured candidates; stringify conservatively
            text = "".join([getattr(c, "content", "") for c in text])
        if not isinstance(text, str):
            raise RuntimeError("Model did not return text content")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to salvage JSON by extracting first/last brackets
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                parsed = json.loads(text[start_idx : end_idx + 1])
            else:
                raise

        results: List[Dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            uuid_val = item.get("uuid") or item.get("id") or item.get("video_id")
            reasoning_val = item.get("reasoning") or item.get("rationale")
            if not uuid_val or not reasoning_val:
                continue
            results.append({"uuid": str(uuid_val), "reasoning": str(reasoning_val).strip()})

        # Enforce max_items cap client-side as well
        return results[:max_items]

    def extract_memories(
        self,
        *,
        start_time_iso: Optional[str] = None,
        end_time_iso: Optional[str] = None,
        limit: Optional[int] = None,
        max_items: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Full pipeline: fetch candidates (time-filtered), call model, return JSON-ready list.
        """
        print("🧩 Starting memory extraction pipeline...")
        candidates = self.fetch_annotated_videos(
            start_time_iso=start_time_iso, end_time_iso=end_time_iso, limit=limit
        )
        # Map annotations by uuid for later augmentation
        uuid_to_annotation: Dict[str, str] = {str(c["uuid"]): str(c.get("annotation", "")) for c in candidates}
        print(f"🗂️ Selecting up to {max_items} memories via Gemini...")
        selected = self.select_memories(candidates, max_items=max_items)
        try:
            print(f"✅ Selected {len(selected)} memories")
        except Exception:
            pass
        # Attach annotations to selected results for richer chat context
        for item in selected:
            ann = uuid_to_annotation.get(item.get("uuid", ""), "")
            if ann:
                item["annotation"] = ann
        # Cache for downstream chat/context usage
        self.memories_context = selected
        # Prepare actual video context via VideoClip utilities
        try:
            print("📦 Preparing video context (uploading to Gemini for LangChain media parts)...")
            self._prepare_video_context_with_videoclips([item["uuid"] for item in selected])
        except Exception as e:
            self.video_files_context = []
            print(f"⚠️ Failed to prepare video context; proceeding without videos: {e}")
        return selected

    def _prepare_video_context_with_videoclips(self, uuids: Sequence[str]) -> None:
        uploaded: List[Any] = []
        lc_parts: List[Dict[str, Any]] = []
        try:
            print(f"⬆️ Uploading {len(list(uuids))} videos to Gemini...")
        except Exception:
            pass
        for uuid_val in uuids:
            try:
                clip = VideoClip(uuid_val)
                # Prepare Gemini file for non-LangChain direct calls (kept for compatibility)
                tmp_path = clip.download_to_tempfile()
                try:
                    file_handle = clip.upload_to_gemini(tmp_path)
                    uploaded.append(file_handle)
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception:
                        pass
                # Prepare LangChain media part using uploaded file URI/name
                file_uri = getattr(file_handle, "uri", None) or getattr(file_handle, "name", None)
                if file_uri:
                    lc_parts.append({
                        "type": "media",
                        "mime_type": "video/mp4",
                        "file_uri": file_uri,
                    })
                try:
                    print(f"   ✅ Prepared media part for {uuid_val}")
                except Exception:
                    pass
            except Exception as e:
                # Skip failures to keep best-effort context
                try:
                    print(f"   ❌ Skipping {uuid_val}: {e}")
                except Exception:
                    pass
                continue
        self.video_files_context = uploaded
        # Replace any previous cache
        # Note: temp paths will be cleaned after chat loop ends or on agent reinit
        self._langchain_media_parts = lc_parts
        try:
            count = len(lc_parts)
            sample = (lc_parts[0].get("file_uri") if count else None)
            print(f"✅ Prepared {count} video media parts for LangChain. Sample URI: {sample}")
        except Exception:
            pass

    # -------------------------
    # Therapist chat via LangChain
    # -------------------------

    def _build_system_prompt(self) -> str:
        """
        Create the ESI therapist system prompt describing role, goals, and style.
        """
        return (
            "You are an expert Episodic Specificity Induction (ESI) therapist helping a patient with early-stage Alzheimer's prepare for a Subsequent Retrieval (SR) session. "
            "Your aims: (1) gently cue vivid, specific memories; (2) scaffold sensory detail (sight, sound, touch, smell, taste), spatial/temporal anchors, and social/goal context; (3) cultivate safety and agency; (4) keep responses concise and momentum-building.\n\n"
            "Therapeutic style: warm, validating, non-judgmental, collaborative. Ask one clear question at a time. Encourage but never pressure. If distress surfaces, acknowledge it, downshift pace, and offer grounding (breath, present-moment sensory check).\n\n"
            "Context: the appropriate memory video clips have been provided as attached media in this conversation. Use them as gentle cues to probe vivid, specific recall. Do not force the content of any clip; let the patient lead and notice what stands out for them. Avoid narrating the whole clip—ask targeted questions that invite sensory and spatiotemporal detail.\n\n"
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

    def _ensure_session_saved(self) -> None:
        if self.session is None:
            raise ValueError("Chat session is not initialized. Provide session or session_id to ESIAgent.")
        try:
            # Best-effort: safe to call even if already exists; ignore failure
            self.session.save_to_supabase()
        except Exception:
            pass

    def chat(self, user_text: str) -> str:
        """
        Generate a therapist reply given user input, persisting to Supabase-backed chat history.
        Requires LangChain + Gemini and an initialized ChatSession.
        """
        if ChatGoogleGenerativeAI is None or SystemMessage is None:
            raise RuntimeError("LangChain Google GenAI dependencies are not available. Install langchain-google-genai.")
        if self.llm is None:
            raise RuntimeError("LLM not initialized")
        self._ensure_session_saved()

        # Load prior messages
        history_msgs: List[Any] = []
        try:
            assert self.session is not None
            prior = self.session.get_chat_messages()
        except Exception:
            prior = []
        for m in prior:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "")
            if role == "assistant":
                history_msgs.append(AIMessage(content=content))
            else:
                history_msgs.append(HumanMessage(content=content))

        system_msg = SystemMessage(content=self._build_system_prompt())

        # Build messages based on context mode
        if self._langchain_media_parts:
            # Compose turn with attached videos
            user_content: List[Dict[str, Any]] = []
            user_content.append({"type": "text", "text": "Use attached clips as gentle cues (do not force)."})
            for part in self._langchain_media_parts:
                user_content.append(part)
            user_content.append({"type": "text", "text": user_text})
            try:
                print(f"📎 Attaching {len(self._langchain_media_parts)} video media parts to this chat turn")
            except Exception:
                pass
            messages: List[Any] = [system_msg] + history_msgs + [HumanMessage(content=user_content)]
        else:
            # Fallback to textual memory annotations context
            raise ValueError("No video media parts available for therapist chat")

        # Invoke LLM
        try:
            print("🧠 Invoking Gemini chat model...")
        except Exception:
            pass
        ai_response = self.llm.invoke(messages)
        try:
            print("✅ Received model reply")
        except Exception:
            pass
        ai_text: str = getattr(ai_response, "content", "") if ai_response else ""

        # Persist messages
        if ChatMessage is None:
            raise RuntimeError("ChatMessage module not available for persistence")
        assert self.session is not None
        try:
            ChatMessage(content=user_text, session_id=self.session.id, role="user").save_to_supabase()
        except Exception:
            pass
        try:
            ChatMessage(content=ai_text, session_id=self.session.id, role="assistant").save_to_supabase()
        except Exception:
            pass

        return ai_text


    def kickoff(self) -> str:
        """
        Start the session with the therapist speaking first.
        Uses attached video clips (if available) and asks the first gentle recall question.
        Persists only the assistant message (no synthetic user prompt is saved).
        """
        if ChatGoogleGenerativeAI is None or SystemMessage is None:
            raise RuntimeError("LangChain Google GenAI dependencies are not available. Install langchain-google-genai.")
        if self.llm is None:
            raise RuntimeError("LLM not initialized")
        self._ensure_session_saved()

        system_msg = SystemMessage(content=self._build_system_prompt())

        if not self._langchain_media_parts:
            raise ValueError("No video media parts available for therapist kickoff")

        kickoff_text = (
            "Please open the session with a warm, brief greeting and ask ONE gentle, concrete recall "
            "question grounded in the attached clips. Avoid narrating the clips. Invite a specific, sensory detail "
            "(e.g., lighting, a sound, an object, or who was there)."
        )
        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "text": "Use attached clips as gentle cues (do not force)."})
        for part in self._langchain_media_parts:
            user_content.append(part)
        # Add the kickoff directive as text so the model initiates
        user_content.append({"type": "text", "text": kickoff_text})

        try:
            print(f"🟢 Kickoff: attaching {len(self._langchain_media_parts)} video media parts and requesting therapist to start")
        except Exception:
            pass

        messages: List[Any] = [system_msg, HumanMessage(content=user_content)]
        ai_response = self.llm.invoke(messages)
        ai_text: str = getattr(ai_response, "content", "") if ai_response else ""

        # Persist only assistant kickoff message
        if ChatMessage is None:
            raise RuntimeError("ChatMessage module not available for persistence")
        assert self.session is not None
        try:
            ChatMessage(content=ai_text, session_id=self.session.id, role="assistant").save_to_supabase()
        except Exception:
            pass

        return ai_text


def _parse_iso_or_none(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    # Validate basic ISO-8601; accept as-is if parseable
    try:
        # Permit date-only or datetime
        if len(value) <= 10:
            datetime.fromisoformat(value)
        else:
            # fromisoformat accepts many variants; this is a sanity check only
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except ValueError:
        raise ValueError(
            "Invalid ISO timestamp. Use YYYY-MM-DD or full ISO-8601 like 2025-09-13T10:00:00Z"
        )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ESI memory extractor and therapist chat")
    parser.add_argument("--start", dest="start", help="ISO start time (inclusive)")
    parser.add_argument("--end", dest="end", help="ISO end time (inclusive)")
    parser.add_argument("--limit", dest="limit", type=int, help="Max candidates to fetch before selection")
    parser.add_argument(
        "--max", dest="max_items", type=int, default=10, help="Max items to return after selection"
    )
    parser.add_argument(
        "--model", dest="model_name", default="gemini-2.5-flash", help="Gemini model name"
    )
    parser.add_argument(
        "--table", dest="table_name", default="test_videos", help="Supabase table holding videos"
    )
    parser.add_argument(
        "--ts-col", dest="timestamp_column", default="time_created", help="Timestamp column for filtering"
    )
    # Chat mode
    parser.add_argument("--session-id", dest="session_id", help="Chat session UUID for persistence")
    parser.add_argument("--chat", dest="chat_text", help="Send a single user message and print therapist reply")
    parser.add_argument("--interactive", action="store_true", default=True, help="Start an interactive ESI therapist chat loop")

    args = parser.parse_args()

    start_iso = _parse_iso_or_none(args.start)
    end_iso = _parse_iso_or_none(args.end)

    agent = ESIAgent(session_id=args.session_id)
    # Apply CLI overrides
    agent.model_name = args.model_name
    agent.table_name = args.table_name
    agent.timestamp_column = args.timestamp_column
    # Reinitialize LLM with updated model if available
    try:
        if ChatGoogleGenerativeAI is not None:
            agent.llm = ChatGoogleGenerativeAI(
                model=agent.model_name,
                google_api_key=agent.gemini_api_key,
            )
    except Exception:
        pass

    results = agent.extract_memories(
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        limit=args.limit,
        max_items=args.max_items,
    )

    # Always print the selected memories (JSON array)
    print(json.dumps(results, ensure_ascii=False))

    # Optionally run chat
    if args.chat_text:
        reply = agent.chat(args.chat_text)
        print("\n--- Therapist ---\n" + reply)
    elif args.interactive:
        print("\nEntering interactive ESI therapist mode. Ctrl+C to exit.\n")
        try:
            # Therapist starts the session
            try:
                first = agent.kickoff()
                print("Therapist:", first)
            except Exception as e:
                print(f"⚠️ Kickoff failed: {e}")
            while True:
                user_text = input("You: ").strip()
                if not user_text:
                    continue
                reply = agent.chat(user_text)
                print("Therapist:", reply)
        except KeyboardInterrupt:
            print("\nGoodbye.")


if __name__ == "__main__":
    main()


