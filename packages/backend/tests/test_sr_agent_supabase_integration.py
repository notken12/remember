#!/usr/bin/env python3

import os
import sys
import unittest
import asyncio
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from sr_agent import SRAgentRunner


@unittest.skipUnless(
    all([os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"), os.getenv("DATABASE_URL"), os.getenv("GEMINI_API_KEY")]),
    "Real Supabase/Gemini credentials required for integration test",
)
class TestSupabaseIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_kickoff_and_first_prompt_from_supabase(self):
        # Ensure we rely on Supabase discovery (clear fallback)
        os.environ.pop("SR_FALLBACK_CLIP_IDS", None)
        runner = SRAgentRunner(session_id="sr_int_test")

        received: List[str] = []
        async for part in runner.kickoff():
            received.append(str(part))
            if len(received) > 5:
                break
        self.assertGreaterEqual(len(received), 1)


if __name__ == "__main__":
    unittest.main()


