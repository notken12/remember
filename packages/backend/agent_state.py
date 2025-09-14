from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import add_messages

from supabase_client import supabase


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    session_id: str


def json_to_message(json: dict):
    message_type = json.get("type")
    if message_type == "human":
        return HumanMessage(content=json["content"], **json)
    elif message_type == "ai":
        return AIMessage(content=json["content"], **json)
    elif message_type == "tool":
        return ToolMessage(content=json["content"], **json)
    else:
        raise ValueError(f"Unknown message type: {message_type}. Message: {json}")


def get_state_from_supabase(session_id: str) -> State:
    messages = (
        supabase.table("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )
    return State(
        messages=[json_to_message(msg["data"]) for msg in messages.data],
        session_id=session_id,
    )
