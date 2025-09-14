from fastapi import FastAPI, Request
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
async def web_stream():
    return EventSourceResponse(event_generator())


def event_generator():
    pubsub = r.pubsub()
    pubsub.subscribe("web_stream")
    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=None)
        if message is not None:
            yield ServerSentEvent(data=message["data"])


@app.post("/current_transcription")
async def set_current_transcription(text: str):
    msg = json.dumps({"type": "data-transcription", "data": {"text": text}})
    r.publish("web_stream", msg)
    return
