from io import IOBase
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Dict, Optional, Union, Tuple, Literal, Any

class FunctionParameter(BaseModel):
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str] = None
    defs: Dict[str, Any] = Field(default=None, alias="$defs")

    class Config:
        population_by_name = True

class Function(BaseModel):
    name: str
    description: str = Field(default=None)
    parameters: Optional[FunctionParameter] = Field(default=None, exclude=None)

class Tool(BaseModel):
    type: str
    function: Function

    @classmethod
    def parse_raw(cls, json_str: str) -> 'Tool':
        """从JSON字符串解析Tool对象"""
        return cls.model_validate_json(json_str)

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    name: Optional[str] = None
    content: Optional[Union[str, List[ContentItem]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    class Config:
        extra = "allow"  # 允许额外的字段

class FunctionChoice(BaseModel):
    name: str

class ToolChoice(BaseModel):
    type: str
    function: Optional[FunctionChoice] = None

class BaseRequest(BaseModel):
    request_type: Optional[Literal["chat", "image", "audio", "moderation"]] = Field(default=None, exclude=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*shadows an attribute.*")

class JsonSchema(BaseModel):
    name: str
    schema: Dict[str, Any] = Field(validation_alias='schema')

    model_config = ConfigDict(protected_namespaces=())

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchema] = None

class Thinking(BaseModel):
    budget_tokens: Optional[int] = None
    type: Optional[Literal["enabled", "disabled"]] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None

class RequestModel(BaseRequest):
    model: str
    messages: List[Message]
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stream: Optional[bool] = None
    include_usage: Optional[bool] = None
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    user: Optional[str] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    tools: Optional[List[Tool]] = None
    response_format: Optional[ResponseFormat] = None
    thinking: Optional[Thinking] = None
    stream_options: Optional[StreamOptions] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    def get_last_text_message(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.content:
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list):
                    for item in reversed(message.content):
                        if item.type == "text" and item.text:
                            return item.text
        return ""

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        # 检查并处理 tools 字段
        if 'tools' in data and data['tools']:
            for tool in data['tools']:
                if 'function' in tool:
                    function_data = tool['function']
                    # 如果 parameters 为空或没有 properties，则移除
                    if 'parameters' in function_data and (
                        function_data['parameters'] is None or
                        not function_data['parameters'].get('properties')
                    ):
                        function_data.pop('parameters', None)

        return data

class ImageGenerationRequest(BaseRequest):
    prompt: str
    model: Optional[str] = "dall-e-3"
    n:  Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    stream: bool = False

class EmbeddingRequest(BaseRequest):
    input: Union[str, List[Union[str, int, List[int]]]]  # 支持字符串或数组
    model: str
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    stream: bool = False

class AudioTranscriptionRequest(BaseRequest):
    file: Tuple[str, IOBase, str]
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    stream: bool = False
    timestamp_granularities: Optional[List[str]] = Field(default=["segment"])

    class Config:
        arbitrary_types_allowed = True

class ModerationRequest(BaseRequest):
    input: Union[str, List[str]]
    model: Optional[str] = "text-moderation-latest"
    stream: bool = False

class TextToSpeechRequest(BaseRequest):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False  # Add this line

class UnifiedRequest(BaseModel):
    data: Union[RequestModel, ImageGenerationRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest, TextToSpeechRequest]

    @model_validator(mode='before')
    @classmethod
    def set_request_type(cls, values):
        if isinstance(values, dict):
            if "messages" in values:
                values["data"] = RequestModel(**values)
                values["data"].request_type = "chat"
            elif "prompt" in values:
                values["data"] = ImageGenerationRequest(**values)
                values["data"].request_type = "image"
            elif "file" in values:
                values["data"] = AudioTranscriptionRequest(**values)
                values["data"].request_type = "audio"
            elif "tts" in values.get("model", ""):
                values["data"] = TextToSpeechRequest(**values)
                values["data"].request_type = "tts"
            elif "text-embedding" in values.get("model", ""):
                values["data"] = EmbeddingRequest(**values)
                values["data"].request_type = "embedding"
            elif "input" in values:
                values["data"] = ModerationRequest(**values)
                values["data"].request_type = "moderation"
            else:
                raise ValueError("无法确定请求类型")
        return values

if __name__ == "__main__":
    # 示例JSON字符串
    json_str = '''
    {
        "type": "function",
        "function": {
            "name": "clock-time____getCurrentTime____standalone",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
    '''

    # 解析JSON字符串为Tool对象
    tool = Tool.parse_raw(json_str)

    # parameters 字段将被自动排除
    print(tool.model_dump(exclude_unset=True))