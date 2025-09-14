#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure backend package is importable
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
    QuestionState,
)


class TestStateQueue(unittest.TestCase):
    def test_enqueue_due_pop_roundtrip(self):
        state = {"sr": {"interval_seconds": [10, 20, 40]}}
        sr = ensure_sr_slice(state)
        base = datetime(2025, 1, 1, 12, 0, 0)
        enqueue(state, clip_id="c1", questions=[{"text_cue": "Q1", "answer": "A1"}], interval_index=0, base_time=base)
        enqueue(state, clip_id="c2", questions=[{"text_cue": "Q1", "answer": "A1"}], interval_index=1, base_time=base)
        self.assertIsNone(get_next_due(state, now=base + timedelta(seconds=9)))
        head = get_next_due(state, now=base + timedelta(seconds=10))
        self.assertIsNotNone(head)
        self.assertEqual(head["clip_id"], "c1")
        popped = pop_next_due(state, now=base + timedelta(seconds=10))
        self.assertEqual(popped["clip_id"], "c1")
        self.assertIsNone(get_next_due(state, now=base + timedelta(seconds=19)))
        head2 = pop_next_due(state, now=base + timedelta(seconds=20))
        self.assertEqual(head2["clip_id"], "c2")


class TestSelectionAndQuestions(unittest.TestCase):
    def test_selection_and_session(self):
        state = {"sr": {"interval_seconds": [30, 60]}}
        sr = ensure_sr_slice(state)
        # select_first_n is trivial; test session flow instead
        base = datetime(2025, 1, 1, 12, 0, 0)
        enqueue(state, clip_id="x", questions=[{"text_cue": "Qx1", "answer": "Ax1"}, {"text_cue": "Qx2", "answer": "Ax2"}], interval_index=0, base_time=base)
        item = pop_next_due(state, now=base + timedelta(seconds=30))
        prompt = begin_session(state, item)
        self.assertEqual(prompt, "Qx1")
        append_answer_and_advance(state, "ok")
        append_answer_and_advance(state, "ok2")
        self.assertTrue(session_finished(state))
        result = evaluate_session(state)
        self.assertTrue(result["success"])  # 2/2 answered
        reschedule(state, success=True, now=base + timedelta(seconds=31))


class TestKickoffFlow(unittest.TestCase):
    def test_kickoff_like_flow(self):
        # Simulate kickoff: select two clips, generate questions, enqueue, due check
        state = {"sr": {"interval_seconds": [5, 10], "questions_per_clip": 1}}
        sr = ensure_sr_slice(state)
        now = datetime.utcnow()
        for cid in ["v1", "v2"]:
            enqueue(state, clip_id=cid, questions=[{"text_cue": f"Q-{cid}", "answer": "A"}], interval_index=0, base_time=now)
        due = get_next_due(state, now=now + timedelta(seconds=6))
        self.assertIsNotNone(due)


if __name__ == "__main__":
    unittest.main()


