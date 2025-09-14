from langchain_core.messages import AIMessageChunk, ToolCallChunk, ToolMessage
from protocol import (
    StreamProtocolPart,
    TextStartPart,
    TextDeltaPart,
    TextEndPart,
    ToolInputStartPart,
    ToolInputDeltaPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
    OutputWithMetadata,
)
from typing import AsyncGenerator, Tuple, List, cast, Any
import json
import uuid


async def parse_langgraph_stream(
    stream: AsyncGenerator[Tuple[AIMessageChunk | ToolMessage, Any], None],
) -> AsyncGenerator[StreamProtocolPart, None]:
    tool_call_chunks: List[ToolCallChunk] = []
    gathered: List[ToolCallChunk] = []
    current_text_part_id: str | None = None
    gathered_messages: List[AIMessageChunk | ToolMessage] = []

    async for chunk, _metadata in stream:
        print(f"Chunk: {chunk}")
        if len(gathered_messages) == 0 or chunk.id != gathered_messages[-1].id:
            if len(gathered_messages) > 0:
                message = gathered_messages[-1]
                # yield message
            gathered_messages.append(chunk)
        else:
            gathered_messages[-1] = cast(
                AIMessageChunk | ToolMessage, gathered_messages[-1] + chunk
            )

        if isinstance(chunk, AIMessageChunk):
            content = chunk.content
            # Handle case where content is a list of dictionaries (new format)
            if isinstance(content, list):
                # Extract text from list of content blocks
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content

            if content:
                if not current_text_part_id:
                    current_text_part_id = str(uuid.uuid4())
                    yield TextStartPart(
                        id=current_text_part_id,
                    )
                yield TextDeltaPart(
                    id=current_text_part_id,
                    delta=str(content),
                )
            else:
                if current_text_part_id:
                    yield TextEndPart(
                        id=current_text_part_id,
                    )
                    current_text_part_id = None

            for tool_call_chunk in chunk.tool_call_chunks:
                if tool_call_chunk["id"]:
                    gathered.append(tool_call_chunk)
                    yield ToolInputStartPart(
                        toolCallId=tool_call_chunk["id"],
                        toolName=tool_call_chunk["name"] or "",
                    )
                else:
                    if not gathered[-1]["id"]:
                        continue
                    gathered[-1]["args"] = (gathered[-1]["args"] or "") + (
                        tool_call_chunk["args"] or ""
                    )
                    # Check if this tool should stream deltas
                    tool_name = gathered[-1].get("name") or ""
                    yield ToolInputDeltaPart(
                        toolCallId=gathered[-1]["id"],
                        inputTextDelta=tool_call_chunk["args"] or "",
                    )

            if len(chunk.tool_call_chunks) == 0 and len(gathered) > 0:
                last_tool_call = gathered.pop()
                if last_tool_call["id"] and last_tool_call["args"]:
                    yield ToolInputAvailablePart(
                        toolCallId=last_tool_call["id"],
                        toolName=last_tool_call["name"] or "",
                        input=json.loads(last_tool_call["args"]),
                    )
                    tool_call_chunks.append(last_tool_call)
        else:
            # This is a ToolMessage - the result of a tool execution
            json_str = cast(str, chunk.content)
            try:
                json_str = json.loads(json_str)
            except Exception as e:
                pass

            tool_call_id = getattr(chunk, "tool_call_id", "unknown")
            artifact = getattr(chunk, "artifact", None)

            tool_output_part = ToolOutputAvailablePart(
                toolCallId=tool_call_id,
                output=OutputWithMetadata(
                    result=json_str,
                    metadata=artifact,
                ),
            )
            yield tool_output_part

    if current_text_part_id:
        yield TextEndPart(
            id=current_text_part_id,
        )

    if len(gathered_messages) > 0:
        message = gathered_messages[-1]
        # yield message
