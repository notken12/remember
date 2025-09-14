#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from sr_agent import (
    ensure_sr_slice,
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


class TestChatFlow(unittest.TestCase):
    def setUp(self):
        self.state = {"sr": {"interval_seconds": [2, 4]}}
        sr = ensure_sr_slice(self.state)
        # Enqueue clip with two questions
        base = datetime(2025, 1, 1, 12, 0, 0)
        enqueue(
            self.state,
            clip_id="clipA",
            questions=[{"text_cue": "Q1?", "answer": "A1"}, {"text_cue": "Q2?", "answer": "A2"}],
            interval_index=0,
            base_time=base,
        )

    def test_due_session_and_progression(self):
        # No active session initially; nothing due at t+1s
        # Not due at 12:00:01
        self.assertIsNone(get_next_due(self.state, now=datetime(2025, 1, 1, 12, 0, 1)))
        # Due at 12:00:02
        item = pop_next_due(self.state, now=datetime(2025, 1, 1, 12, 0, 2))
        prompt = begin_session(self.state, item)
        self.assertEqual(prompt, "Q1?")
        append_answer_and_advance(self.state, "first answer")
        self.assertFalse(session_finished(self.state))
        self.assertEqual(current_prompt(self.state), "Q2?")
        append_answer_and_advance(self.state, "second answer")
        self.assertTrue(session_finished(self.state))
        result = evaluate_session(self.state)
        self.assertTrue(result["success"])  # answered both
        reschedule(self.state, success=True, now=datetime(2025, 1, 1, 12, 0, 3))


if __name__ == "__main__":
    unittest.main()


