from typing import Any, Dict, Literal, Union, Optional

from pydantic import BaseModel


class MessageStartPart(BaseModel):
    type: Literal["start"] = "start"
    messageId: str


class TextStartPart(BaseModel):
    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaPart(BaseModel):
    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndPart(BaseModel):
    type: Literal["text-end"] = "text-end"
    id: str


class ReasoningStartPart(BaseModel):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class ReasoningDeltaPart(BaseModel):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    id: str
    delta: str


class ReasoningEndPart(BaseModel):
    type: Literal["reasoning-end"] = "reasoning-end"
    id: str


class FinishPart(BaseModel):
    type: Literal["finish"] = "finish"
    id: str


class ErrorPart(BaseModel):
    type: Literal["error"] = "error"
    errorText: str


class ToolInputStartPart(BaseModel):
    type: Literal["tool-input-start"] = "tool-input-start"
    toolCallId: str
    toolName: str


class ToolInputDeltaPart(BaseModel):
    type: Literal["tool-input-delta"] = "tool-input-delta"
    toolCallId: str
    inputTextDelta: str


class ToolInputAvailablePart(BaseModel):
    type: Literal["tool-input-available"] = "tool-input-available"
    toolCallId: str
    toolName: str
    input: Dict[str, Any]


class OutputWithMetadata(BaseModel):
    result: Any
    metadata: Any


class ToolOutputAvailablePart(BaseModel):
    type: Literal["tool-output-available"] = "tool-output-available"
    toolCallId: str
    output: OutputWithMetadata


class FinishMessagePart(BaseModel):
    type: Literal["finish"] = "finish"


class StatusUpdatePart(BaseModel):
    type: Literal["status-update"] = "status-update"
    status: str
    timestamp: str


class ResumeStartPart(BaseModel):
    type: Literal["resume-start"] = "resume-start"
    timestamp: str


class HumanInterruptPart(BaseModel):
    type: Literal["data-human-interrupt"] = "data-human-interrupt"
    id: Optional[str] = None
    data: Dict[str, Any]  # Will contain {content, timestamp, files}


class WorkflowCancelPart(BaseModel):
    type: Literal["data-workflow-cancel"] = "data-workflow-cancel"
    timestamp: str
    reason: Optional[str] = None


StreamTerminationPart = Literal["[DONE]"]

StreamProtocolPart = Union[
    MessageStartPart,
    TextStartPart,
    TextDeltaPart,
    TextEndPart,
    ReasoningStartPart,
    ReasoningDeltaPart,
    ReasoningEndPart,
    FinishPart,
    ErrorPart,
    ToolInputStartPart,
    ToolInputDeltaPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
    FinishMessagePart,
    StatusUpdatePart,
    HumanInterruptPart,
    WorkflowCancelPart,
    StreamTerminationPart,
    ResumeStartPart,
]
