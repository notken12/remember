#!/usr/bin/env python3

import os
import sys
import unittest
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from SRAgent import SRAgent, AgentGraphState
from Question import Question


class TestChatFlow(unittest.TestCase):
    def setUp(self):
        self.agent = SRAgent(interval_seconds=[2, 4], questions_per_clip=2)
        self.agent.set_graph_state({"sr": {}})  # type: ignore

        # Seed a selected clip and questions
        self.agent._selected_clips = ["clipA"]
        self.agent._clip_questions = {
            "clipA": [
                Question(video_clip=None, text_cue="Q1?", answer="A1"),
                Question(video_clip=None, text_cue="Q2?", answer="A2"),
            ]
        }
        base = datetime(2025, 1, 1, 12, 0, 0)
        self.agent.enqueue_clip_for_spaced_retrieval(
            "clipA", self.agent._clip_questions["clipA"], interval_index=0, base_time=base
        )

    def test_due_session_and_progression(self):
        # No active session initially; nothing due at t+1s
        msg_idle = self.agent.chat("hi")
        self.assertTrue(isinstance(msg_idle, str))

        # At t+2s, should start session
        start_msg = self.agent.conduct_clip_session(
            self.agent.pop_next_due_item(now=datetime(2025, 1, 1, 12, 0, 2))  # type: ignore
        )
        self.assertIn("Let's focus on this moment", start_msg)

        # First answer advances q_index and returns next prompt
        reply = self.agent.chat("first answer")
        self.assertIn("Next one:", reply)

        # Second answer concludes and reschedules
        reply2 = self.agent.chat("second answer")
        self.assertIn("circle back", reply2)


if __name__ == "__main__":
    unittest.main()


