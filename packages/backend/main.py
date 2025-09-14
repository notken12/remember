import asyncio
import uuid
from fastapi import FastAPI, Request, Response
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse
import json
import redis
from fastapi.middleware.cors import CORSMiddleware

from esi_agent import chat, kickoff
from sr_agent import SRAgentRunner
from AssistantAgent import AssistantAgent
from protocol import FinishMessagePart, MessageStartPart

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


@app.post("/esi_session")
async def start_esi_session(session_id: str):
    async def generator():
        start_part = MessageStartPart(messageId=str(uuid.uuid4())).model_dump_json()
        r.publish("web_stream", start_part)
        yield ServerSentEvent(data=start_part)

        async for part in kickoff(session_id):
            r.publish("web_stream", part.model_dump_json())
            yield ServerSentEvent(data=part.model_dump_json())
        finish_part = FinishMessagePart()
        r.publish("web_stream", finish_part.model_dump_json())
        yield ServerSentEvent(data=finish_part.model_dump_json())

    return EventSourceResponse(generator())


@app.post("/esi_chat")
async def esi_chat(session_id: str, user_message: str):
    set_current_transcription(user_message, is_final=True)

    async def generator():
        start_part = MessageStartPart(messageId=str(uuid.uuid4())).model_dump_json()
        r.publish("web_stream", start_part)
        yield ServerSentEvent(data=start_part)

        async for part in chat(session_id, user_message):
            r.publish("web_stream", part.model_dump_json())
            yield ServerSentEvent(data=part.model_dump_json())
        finish_part = FinishMessagePart()
        r.publish("web_stream", finish_part.model_dump_json())
        yield ServerSentEvent(data=finish_part.model_dump_json())

    return EventSourceResponse(generator())


@app.post("/assistant_chat")
async def assistant_chat(user_message: str):
    assistant_agent = AssistantAgent()

    async def generator():
        start_part = MessageStartPart(messageId=str(uuid.uuid4())).model_dump_json()
        r.publish("web_stream", start_part)
        yield ServerSentEvent(data=start_part)

        async for part in assistant_agent.query(user_message):
            r.publish("web_stream", part.model_dump_json())
            yield ServerSentEvent(data=part.model_dump_json())
        finish_part = FinishMessagePart()
        r.publish("web_stream", finish_part.model_dump_json())
        yield ServerSentEvent(data=finish_part.model_dump_json())

    return EventSourceResponse(generator())


@app.post("/sr_session")
async def start_sr_session(session_id: str):
    runner = SRAgentRunner(session_id=session_id)

    async def generator():
        start_part = MessageStartPart(messageId=str(uuid.uuid4())).model_dump_json()
        r.publish("web_stream", start_part)
        yield ServerSentEvent(data=start_part)

        async for part in runner.kickoff():
            r.publish("web_stream", part.model_dump_json())
            yield ServerSentEvent(data=part.model_dump_json())
        finish_part = FinishMessagePart()
        r.publish("web_stream", finish_part.model_dump_json())
        yield ServerSentEvent(data=finish_part.model_dump_json())

    return EventSourceResponse(generator())


@app.post("/sr_chat")
async def sr_chat(session_id: str, user_message: str):
    set_current_transcription(user_message, is_final=True)
    runner = SRAgentRunner(session_id=session_id)

    async def generator():
        start_part = MessageStartPart(messageId=str(uuid.uuid4())).model_dump_json()
        r.publish("web_stream", start_part)
        yield ServerSentEvent(data=start_part)

        async for part in runner.chat(user_message):
            r.publish("web_stream", part.model_dump_json())
            yield ServerSentEvent(data=part.model_dump_json())
        finish_part = FinishMessagePart()
        r.publish("web_stream", finish_part.model_dump_json())
        yield ServerSentEvent(data=finish_part.model_dump_json())

    return EventSourceResponse(generator())


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
