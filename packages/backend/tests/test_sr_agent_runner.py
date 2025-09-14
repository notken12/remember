#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime, timedelta
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from sr_agent import (
    SRAgentRunner,
    ensure_sr_slice,
    configure_sr_from_env,
    load_recent_video_ids,
    select_first_n,
    enqueue,
    get_next_due,
    pop_next_due,
    begin_session,
    current_prompt,
    append_answer_and_advance,
    session_finished,
    evaluate_session,
    reschedule,
)


class TestSRRunnerStateHelpers(unittest.TestCase):
    def test_schedule_and_session_flow(self):
        state = {"sr": {}}
        sr = ensure_sr_slice(state)
        configure_sr_from_env(state)

        # Enqueue one clip with two questions
        enqueue(
            state,
            clip_id="clip1",
            questions=[{"text_cue": "Q1?", "answer": "A1"}, {"text_cue": "Q2?", "answer": "A2"}],
            interval_index=0,
            base_time=datetime(2025, 1, 1, 12, 0, 0),
        )

        # Not due at 11:59:59
        self.assertIsNone(get_next_due(state, now=datetime(2025, 1, 1, 11, 59, 59)))
        # Due at 12:00:30 (default 30s step)
        due = get_next_due(state, now=datetime(2025, 1, 1, 12, 0, 30))
        self.assertIsNotNone(due)
        item = pop_next_due(state, now=datetime(2025, 1, 1, 12, 0, 30))
        first = begin_session(state, item)
        self.assertEqual(first, "Q1?")
        self.assertEqual(current_prompt(state), "Q1?")

        # Answer first, advance to Q2
        append_answer_and_advance(state, "first answer")
        self.assertFalse(session_finished(state))
        self.assertEqual(current_prompt(state), "Q2?")

        # Answer second, session finished
        append_answer_and_advance(state, "second answer")
        self.assertTrue(session_finished(state))
        result = evaluate_session(state)
        self.assertTrue(result["success"])  # 2/2 answered
        # Reschedule
        reschedule(state, success=True, now=datetime(2025, 1, 1, 12, 0, 31))
        self.assertIsNone(ensure_sr_slice(state).get("active_session"))


class TestSRRunnerKickoffChat(unittest.IsolatedAsyncioTestCase):
    async def test_kickoff_and_chat_smoke(self):
        # Use a unique session ID; rely on SR_FALLBACK_CLIP_IDS to avoid Supabase in test env
        os.environ["SR_FALLBACK_CLIP_IDS"] = "clipX,clipY"
        os.environ["SR_MAX_CLIPS"] = "1"
        runner = SRAgentRunner(session_id="test_sr_session")

        # Kickoff stream should yield at least one part
        received = []
        async for part in runner.kickoff():
            received.append(part)
            if len(received) > 3:
                break
        self.assertTrue(len(received) >= 1)

        # Chat a single answer; ensure we get a streamed response
        received2 = []
        async for part in runner.chat("My answer"):
            received2.append(part)
            if len(received2) > 3:
                break
        self.assertTrue(len(received2) >= 1)


if __name__ == "__main__":
    unittest.main()


