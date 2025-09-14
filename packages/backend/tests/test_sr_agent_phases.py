#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from sr_agent import (
    SRAgentRunner,
    ensure_sr_slice,
    set_stage,
    get_stage,
    enqueue,
    begin_session,
    append_answer_and_advance,
    evaluate_session,
)


class TestStageHelpers(unittest.TestCase):
    def test_stage_set_and_get(self):
        state = {"sr": {}}
        sr = ensure_sr_slice(state)
        self.assertEqual(get_stage(state), "idle")
        changed = set_stage(state, "kickoff")
        self.assertTrue(changed)
        self.assertEqual(get_stage(state), "kickoff")
        # Setting to same should return False (no change)
        self.assertFalse(set_stage(state, "kickoff"))


class TestAssessmentScoring(unittest.TestCase):
    def test_assessment_priority_over_nonempty(self):
        state = {"sr": {}}
        sr = ensure_sr_slice(state)
        # Two-question session
        enqueue(
            state,
            clip_id="c1",
            questions=[{"text_cue": "Q1", "answer": "A1"}, {"text_cue": "Q2", "answer": "A2"}],
            interval_index=0,
            base_time=datetime(2025, 1, 1, 12, 0, 0),
        )
        item = sr["queue"].pop(0)
        first = begin_session(state, item)
        append_answer_and_advance(state, "some answer")
        append_answer_and_advance(state, "some answer 2")

        # Mark first incorrect, second correct
        sess = ensure_sr_slice(state).get("active_session")
        self.assertIsInstance(sess, dict)
        exchanges = list(sess.get("exchanges", []))
        exchanges[0]["assessment"] = "incorrect"
        exchanges[1]["assessment"] = "correct"
        sess["exchanges"] = exchanges
        ensure_sr_slice(state)["active_session"] = sess

        result = evaluate_session(state)
        self.assertAlmostEqual(result["score"], 0.5, places=3)
        self.assertFalse(result["success"])  # 0.5 < 0.7 threshold

        # Make both correct
        exchanges[0]["assessment"] = "correct"
        sess["exchanges"] = exchanges
        ensure_sr_slice(state)["active_session"] = sess
        result2 = evaluate_session(state)
        self.assertTrue(result2["success"])  # 1.0 >= 0.7


class TestRunnerFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Use fallback to avoid depending on Supabase for discovery
        os.environ["SR_FALLBACK_CLIP_IDS"] = "fake1,fake2"
        os.environ["SR_MAX_CLIPS"] = "1"
        self.runner = SRAgentRunner(session_id="test_sr_phase_flow")

    async def test_fast_start_and_media_once(self):
        # Kickoff should attach media once and start first question (fast-start)
        parts = []
        async for part in self.runner.kickoff():
            parts.append(part)
            if len(parts) > 2:
                break

        state = self.runner._get_state()
        sr = ensure_sr_slice(state)
        # Fast-start implies session_active
        self.assertEqual(sr.get("stage"), "session_active")
        # Media attached list should match selected (size 1 due to SR_MAX_CLIPS)
        self.assertEqual(len(sr.get("media_attached_clips", [])), 1)

        # The messages should contain an early HumanMessage with media blocks
        msgs = state.get("messages", [])
        self.assertGreaterEqual(len(msgs), 2)
        has_media_context = False
        for m in msgs:
            try:
                content = m.content
                if isinstance(content, list) and any(isinstance(x, dict) and x.get("type") == "media" for x in content):
                    has_media_context = True
                    break
            except Exception:
                continue
        self.assertTrue(has_media_context)

        # Now chat a wrong answer, ensure last control message does not reattach media
        parts2 = []
        async for part in self.runner.chat("totally wrong"):
            parts2.append(part)
            if len(parts2) > 2:
                break

        state2 = self.runner._get_state()
        msgs2 = state2.get("messages", [])
        self.assertGreaterEqual(len(msgs2), len(msgs))
        # Check the last HumanMessage added has only text blocks
        last_humans = [m for m in msgs2 if m.__class__.__name__ == "HumanMessage"]
        self.assertGreaterEqual(len(last_humans), 1)
        last = last_humans[-1]
        content = getattr(last, "content", None)
        if isinstance(content, list):
            self.assertFalse(any(isinstance(x, dict) and x.get("type") == "media" for x in content))


if __name__ == "__main__":
    unittest.main()



