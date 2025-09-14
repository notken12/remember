import os
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([supabase_url, supabase_key]):
    raise ValueError(
        "Required environment variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY"
    )

supabase = create_client(supabase_url, supabase_key)
