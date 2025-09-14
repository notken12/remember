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

from SRAgent import (
    SRAgent,
    QueueItem,
    AgentGraphState,
    QuestionState,
)
from Question import Question


class TestQueueItemSerde(unittest.TestCase):
    def test_round_trip(self):
        qs = [Question(video_clip=None, text_cue="What color was the mug?", answer="Blue")]
        now = datetime.utcnow()
        item = QueueItem(clip_id="abc", questions=qs, next_at=now, interval_index=1, attempt_count=2)
        state = item.to_state()
        self.assertEqual(state["clip_id"], "abc")
        self.assertIn("next_at", state)
        rebuilt = QueueItem.from_state(state)
        self.assertEqual(rebuilt.clip_id, item.clip_id)
        self.assertEqual(rebuilt.interval_index, 1)
        self.assertEqual(rebuilt.attempt_count, 2)
        self.assertEqual(len(rebuilt.questions), 1)
        self.assertEqual(rebuilt.questions[0].text_cue, "What color was the mug?")


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.agent = SRAgent(interval_seconds=[10, 20, 40], questions_per_clip=3)

    def test_enqueue_and_peek_pop(self):
        q = [Question(video_clip=None, text_cue="Q1", answer="A1")]
        base = datetime(2025, 1, 1, 12, 0, 0)
        self.agent.enqueue_clip_for_spaced_retrieval("c1", q, interval_index=0, base_time=base)
        self.agent.enqueue_clip_for_spaced_retrieval("c2", q, interval_index=1, base_time=base)

        # Before first due
        self.assertIsNone(self.agent.get_next_due_item(now=base + timedelta(seconds=9)))

        # First becomes due at +10s
        head = self.agent.get_next_due_item(now=base + timedelta(seconds=10))
        self.assertIsNotNone(head)
        self.assertEqual(head.clip_id, "c1")

        popped = self.agent.pop_next_due_item(now=base + timedelta(seconds=10))
        self.assertEqual(popped.clip_id, "c1")

        # Second becomes due at +20s
        self.assertIsNone(self.agent.get_next_due_item(now=base + timedelta(seconds=19)))
        head2 = self.agent.pop_next_due_item(now=base + timedelta(seconds=20))
        self.assertEqual(head2.clip_id, "c2")


class TestSelectionAndQuestions(unittest.TestCase):
    def setUp(self):
        self.agent = SRAgent(interval_seconds=[30, 60], questions_per_clip=2)

    def test_select_clip_subset_deterministic(self):
        all_ids = ["a", "b", "c", "d"]
        subset = self.agent.select_clip_subset(all_ids, max_clips=2)
        self.assertEqual(subset, ["a", "b"])

    def test_prepare_questions_uses_cache(self):
        # Monkeypatch the QuestionGenerator used inside SRAgent
        generated = {
            "x": [Question(video_clip=None, text_cue="Qx1", answer="Ax1"), Question(video_clip=None, text_cue="Qx2", answer="Ax2")],
        }

        class FakeGenerator:
            def __init__(self, clip):
                self.clip = clip
            def generate(self, num_questions: int):
                return generated.get(self.clip.id, [])

        # Inject fake generator and run
        import SRAgent as sr_mod
        sr_mod.QuestionGenerator = FakeGenerator  # type: ignore
        import VideoClip as vc_mod

        class FakeClip:
            def __init__(self, cid):
                self.id = cid

        sr_mod.VideoClip = FakeClip  # type: ignore

        # First call: generates and caches
        result1 = self.agent.prepare_questions_for_clips(["x"], questions_per_clip=2)
        self.assertIn("x", result1)
        self.assertEqual(len(result1["x"]), 2)

        # Second call: should use cache, even if generator returns empty
        generated["x"] = []
        result2 = self.agent.prepare_questions_for_clips(["x"], questions_per_clip=2)
        self.assertIn("x", result2)
        self.assertEqual(len(result2["x"]), 2)


class TestKickoffFlow(unittest.TestCase):
    def setUp(self):
        self.agent = SRAgent(interval_seconds=[5, 10], questions_per_clip=1)
        # Attach empty graph state to avoid None checks
        self.agent.set_graph_state({"sr": {}})  # type: ignore

    def test_kickoff_with_mocked_sources(self):
        # Mock candidate clips and question prep
        self.agent._candidate_limit = 5

        def fake_load(limit=5):
            return ["v1", "v2", "v3"]

        def fake_prepare(ids, questions_per_clip=None):
            return {cid: [Question(video_clip=None, text_cue=f"Q-{cid}", answer="A")] for cid in ids}

        self.agent._load_candidate_clips_from_supabase = fake_load  # type: ignore
        self.agent.prepare_questions_for_clips = fake_prepare  # type: ignore

        msg = self.agent.kickoff(hardcoded_subset_size=2)
        self.assertIn("practice recall for 2", msg)

        # After kickoff, queue should have 2 items scheduled
        due = self.agent.get_next_due_item(now=datetime.utcnow() + timedelta(seconds=6))
        self.assertIsNotNone(due)


if __name__ == "__main__":
    unittest.main()


