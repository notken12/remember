#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime
import asyncio
import uuid

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
)


class TestChatFlow(unittest.TestCase):
    def test_due_session_and_progression(self):
        state = {"sr": {}}
        sr = ensure_sr_slice(state)
        # Enqueue 1 clip with 2 questions due immediately
        enqueue(
            state,
            clip_id="clip1",
            questions=[{"text_cue": "Q1?", "answer": "A1"}, {"text_cue": "Q2?", "answer": "A2"}],
            interval_index=0,
            base_time=datetime(2025, 1, 1, 12, 0, 0),
        )
        # Pop when due
        item = pop_next_due(state, now=datetime(2025, 1, 1, 12, 0, 30))
        self.assertIsNotNone(item)
        first = begin_session(state, item)  # type: ignore[arg-type]
        self.assertEqual(first, "Q1?")

        # Answer Q1
        append_answer_and_advance(state, "answer1")
        self.assertFalse(session_finished(state))
        self.assertEqual(current_prompt(state), "Q2?")

        # Answer Q2; session finished
        append_answer_and_advance(state, "answer2")
        self.assertTrue(session_finished(state))
        result = evaluate_session(state)
        self.assertTrue(result["success"])  # 2/2 non-empty


if __name__ == "__main__":
    unittest.main()


