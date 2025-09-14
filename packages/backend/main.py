import asyncio
from fastapi import FastAPI, Request, Response
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse
import json
import redis
from fastapi.middleware.cors import CORSMiddleware

# Connect to a Redis server running on localhost at the default port (6379)
r = redis.Redis(host="localhost", port=6379, db=0)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/qa_session")
async def start_qa_session(request: Request):
    pass


@app.get("/web_stream")
async def web_stream(response: Response):
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return EventSourceResponse(event_generator())


async def event_generator():
    pubsub = r.pubsub()
    pubsub.subscribe("web_stream")
    while True:
        # Use a short timeout to avoid blocking the event loop
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if message is not None:
            # yield ServerSentEvent(data="hi")
            yield ServerSentEvent(data=message["data"].decode("utf-8"))
        else:
            # Give control back to the event loop when no message is available
            await asyncio.sleep(0.1)


@app.post("/current_transcription")
async def set_current_transcription(text: str, is_final: bool):
    msg = json.dumps(
        {"type": "data-transcription", "data": {"text": text, "is_final": is_final}}
    )
    r.publish("web_stream", msg)
    return
