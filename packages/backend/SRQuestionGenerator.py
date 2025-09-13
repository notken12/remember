#!/usr/bin/env python3

import os
import json
import re
import tempfile
import time
from typing import List, Optional

import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv  # type: ignore

from VideoClip import VideoClip
from Question import Question

# Load environment variables
load_dotenv()


class QuestionGenerator:
    """
    Generate cue-based recall questions for a given video clip using Gemini 2.0 Flash.

    Given a `VideoClip`, this class uploads the underlying video to Gemini, prompts
    with context about cue-based retrieval practice (beneficial for Alzheimer's),
    and parses a strict JSON response into a list of `Question` objects with
    fields `text_cue` and `answer`.
    """

    def __init__(self, video_clip: VideoClip, model_name: str = "gemini-2.5-flash-lite"):
        self.video_clip = video_clip
        self.model_name = model_name
        self.questions: List[Question] = []

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=gemini_api_key)

    def _upload_video_to_gemini(self, video_bytes: bytes):
        """
        Upload bytes as a temporary .mp4 file to Gemini and wait for processing.
        Returns the uploaded file handle usable in `generate_content`.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            video_file = genai.upload_file(path=tmp_path, mime_type="video/mp4")
            while getattr(video_file, "state", None) and getattr(video_file.state, "name", "") == "PROCESSING":
                print("Processing video for questions...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            if getattr(video_file, "state", None) and getattr(video_file.state, "name", "") == "FAILED":
                raise RuntimeError("Video processing failed for question generation")
            return video_file
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _build_prompt(self, desired_count: int) -> str:
        """
        Build the instruction prompt emphasizing cue-based questioning.
        """
        return f"""
You are helping create training materials for individuals with Alzheimer's disease
using retrieval practice. Research shows that cue-based prompts improve long-term
retention: cues should reference distinctive context (when/where), salient events,
and sensory anchors (sounds, colors, objects, emotions). Convert cues into
FULL‑SENTENCE, ANSWERABLE QUESTIONS, for example:
- "At the beginning, when coffee was being served, what made everyone jump?"
- "How did we sing the birthday song—what was unusual about our singing?"
- "What did I bring out to make the party feel more festive?"
- "Which book looked unusual among the others?"
- "What did I pull out when you won the game?"
- "After the accident with the stack of books, what happened next?"

From the provided video, produce {desired_count} concise, specific items for recall
practice. Each item must include:
- text_cue: a single, complete QUESTION sentence (about 10–20 words) that helps
  the person recall a memorable element from the video via context/sensory anchors.
- answer: a short, factual answer grounded in the video content.

Constraints:
- Make cues supportive, concrete, and encouraging; avoid spoilers inside the cue.
- Keep answers brief (1 short phrase or sentence).
- Use present or simple past; avoid speculation.
- Avoid fragments like "Fluffy tail!"; instead ask a clear question such as
  "Which animal's fluffy tail is visible as it plays on the couch?" End with '?'.

Respond with STRICT JSON only, no Markdown, no commentary. Use this schema:
{{
  "video_id": "{self.video_clip.id}",
  "questions": [
    {{ "text_cue": "...", "answer": "..." }}
  ]
}}
"""

    def _extract_json(self, text: str) -> dict:
        """
        Robustly parse JSON from the model response, tolerating accidental fences.
        """
        try:
            return json.loads(text)
        except Exception:
            # Remove code fences if present
            cleaned = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())
            try:
                return json.loads(cleaned)
            except Exception:
                # Try to extract the top-level JSON object heuristically
                obj_match = re.search(r"\{[\s\S]*\}$", cleaned)
                if obj_match:
                    return json.loads(obj_match.group(0))
                # Last resort: extract array and wrap
                arr_match = re.search(r"\[[\s\S]*\]", cleaned)
                if arr_match:
                    return {"video_id": self.video_clip.id, "questions": json.loads(arr_match.group(0))}
                raise

    def generate(self, num_questions: int = 6) -> List[Question]:
        """
        Generate cue-based questions from the associated video clip.

        Returns a list of `Question` objects.
        """
        try:
            # Fetch raw video bytes from Supabase via the VideoClip helper
            video_bytes = self.video_clip._fetch_video_from_supabase()

            # Upload to Gemini and build prompt
            video_file = self._upload_video_to_gemini(video_bytes)
            prompt = self._build_prompt(num_questions)

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json"},
            )

            response = model.generate_content([video_file, prompt])
            payload = self._extract_json(response.text)

            # Support either a dict {{video_id, questions:[...]}} or a bare array
            items = payload.get("questions", payload if isinstance(payload, list) else [])

            self.questions = [
                Question(
                    video_clip=self.video_clip,
                    text_cue=str(item.get("text_cue", "")).strip(),
                    answer=str(item.get("answer", "")).strip(),
                )
                for item in items
                if isinstance(item, dict)
            ]

            print(f"✅ Generated {len(self.questions)} cue-based questions")
            return self.questions
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return []

    def push_all_to_supabase(self) -> bool:
        """
        Push all generated questions to Supabase. Returns True if all succeed.
        """
        if not self.questions:
            print("No questions to push. Call generate() first.")
            return False
        results = [q.push_to_supabase() for q in self.questions]
        success = all(results)
        if success:
            print("🎉 All questions pushed to Supabase successfully")
        else:
            failed = len([r for r in results if not r])
            print(f"⚠️ {failed} question(s) failed to push to Supabase")
        return success


def main():
    """
    Example usage of the QuestionGenerator class.
    """
    # Replace with an actual video UUID from your Supabase database
    video_id = "ade62cd7-3b6c-4e5e-a782-929dab2a2d16"
    clip = VideoClip(video_id)

    generator = QuestionGenerator(clip)
    questions = generator.generate(num_questions=6)

    for i, q in enumerate(questions, start=1):
        print(f"{i}. cue={q.text_cue} | answer={q.answer}")

    # Optionally push all to Supabase
    generator.push_all_to_supabase()


if __name__ == "__main__":
    main()


