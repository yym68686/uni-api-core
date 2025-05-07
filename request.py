import re
import json
import httpx
import base64
import urllib.parse

from .models import RequestModel
from .utils import (
    c3s,
    c3o,
    c3h,
    c35s,
    gemini1,
    gemini2,
    BaseAPI,
    safe_get,
    get_engine,
    get_model_dict,
    get_text_message,
    get_image_message,
)

async def get_gemini_payload(request, engine, provider, api_key=None):
    import re

    headers = {
        'Content-Type': 'application/json'
    }

    # 获取映射后的实际模型ID
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    if request.stream:
        gemini_stream = "streamGenerateContent"
    else:
        gemini_stream = "generateContent"
    url = provider['base_url']
    parsed_url = urllib.parse.urlparse(url)
    if "/v1beta" in parsed_url.path:
        api_version = "v1beta"
    else:
        api_version = "v1"

    url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.split('/models')[0].rstrip('/')}/models/{original_model}:{gemini_stream}?key={api_key}"

    messages = []
    systemInstruction = None
    function_arguments = None
    for msg in request.messages:
        if msg.role == "assistant":
            msg.role = "model"
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = [{"text": msg.content}]
            tool_calls = msg.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_arguments = {
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
            }
            messages.append(
                {
                    "role": "model",
                    "parts": [function_arguments]
                }
            )
        elif msg.role == "tool":
            function_call_name = function_arguments["functionCall"]["name"]
            messages.append(
                {
                    "role": "function",
                    "parts": [{
                    "functionResponse": {
                        "name": function_call_name,
                        "response": {
                            "name": function_call_name,
                            "content": {
                                "result": msg.content,
                            }
                        }
                    }
                    }]
                }
            )
        elif msg.role != "system":
            messages.append({"role": msg.role, "parts": content})
        elif msg.role == "system":
            content[0]["text"] = re.sub(r"_+", "_", content[0]["text"])
            systemInstruction = {"parts": content}

    off_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5", "gemini-2.5-pro"]
    if any(off_model in original_model for off_model in off_models):
        safety_settings = "OFF"
    else:
        safety_settings = "BLOCK_NONE"

    payload = {
        "contents": messages or [{"role": "user", "parts": [{"text": "No messages"}]}],
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                "threshold": "BLOCK_NONE"
            },
        ]
    }

    if systemInstruction:
        if api_version == "v1beta":
            payload["systemInstruction"] = systemInstruction
        if api_version == "v1":
            first_message = safe_get(payload, "contents", 0, "parts", 0, "text", default=None)
            system_instruction = safe_get(systemInstruction, "parts", 0, "text", default=None)
            if first_message and system_instruction:
                payload["contents"][0]["parts"][0]["text"] = system_instruction + "\n" + first_message

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'response_format',
        'stream_options',
    ]
    generation_config = {}

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools" and "gemini-2.0-flash-thinking" in original_model:
                continue
            if field == "tools":
                # 处理每个工具的 function 定义
                processed_tools = []
                for tool in value:
                    function_def = tool["function"]
                    # 处理 parameters.properties 中的 default 字段
                    if safe_get(function_def, "parameters", "properties", default=None):
                        for prop_value in function_def["parameters"]["properties"].values():
                            if "default" in prop_value:
                                # 将 default 值添加到 description 中
                                default_value = prop_value["default"]
                                description = prop_value.get("description", "")
                                prop_value["description"] = f"{description}\nDefault: {default_value}"
                                # 删除 default 字段
                                del prop_value["default"]
                    if function_def["name"] != "googleSearch" and function_def["name"] != "googleSearch":
                        processed_tools.append({"function": function_def})

                if processed_tools:
                    payload.update({
                        "tools": [{
                            "function_declarations": [tool["function"] for tool in processed_tools]
                        }],
                        "tool_config": {
                            "function_calling_config": {
                                "mode": "AUTO"
                            }
                        }
                    })
            elif field == "temperature":
                generation_config["temperature"] = value
            elif field == "max_tokens":
                generation_config["maxOutputTokens"] = value
            elif field == "top_p":
                generation_config["topP"] = value
            else:
                payload[field] = value

    max_token_65k_models = ["gemini-2.5-pro", "gemini-2.0-pro", "gemini-2.0-flash-thinking", "gemini-2.5-flash"]
    payload["generationConfig"] = generation_config
    if "maxOutputTokens" not in generation_config:
        if any(pro_model in original_model for pro_model in max_token_65k_models):
            payload["generationConfig"]["maxOutputTokens"] = 65536
        else:
            payload["generationConfig"]["maxOutputTokens"] = 8192

    # 从请求模型名中检测思考预算设置
    m = re.match(r".*-think-(-?\d+)", request.model)
    if m:
        try:
            val = int(m.group(1))
            if val < 0:
                val = 0
            elif val > 24576:
                val = 24576
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": val}
        except ValueError:
            # 如果转换为整数失败，忽略思考预算设置
            pass

    # 检测search标签
    if request.model.endswith("-search"):
        if "tools" not in payload:
            payload["tools"] = [{"googleSearch": {}}]
        else:
            payload["tools"].append({"googleSearch": {}})

    return url, headers, payload

import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

def create_jwt(client_email, private_key):
    # JWT Header
    header = json.dumps({
        "alg": "RS256",
        "typ": "JWT"
    }).encode()

    # JWT Payload
    now = int(time.time())
    payload = json.dumps({
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now + 3600,
        "iat": now
    }).encode()

    # Encode header and payload
    segments = [
        base64.urlsafe_b64encode(header).rstrip(b'='),
        base64.urlsafe_b64encode(payload).rstrip(b'=')
    ]

    # Create signature
    signing_input = b'.'.join(segments)
    private_key = load_pem_private_key(private_key.encode(), password=None)
    signature = private_key.sign(
        signing_input,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    segments.append(base64.urlsafe_b64encode(signature).rstrip(b'='))
    return b'.'.join(segments).decode()

def get_access_token(client_email, private_key):
    jwt = create_jwt(client_email, private_key)

    with httpx.Client() as client:
        response = client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt
            },
            headers={'Content-Type': "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return response.json()["access_token"]

async def get_vertex_gemini_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    if request.stream:
        gemini_stream = "streamGenerateContent"
    else:
        gemini_stream = "generateContent"
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    search_tool = None

    # https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash?hl=zh-cn
    pro_models = ["gemini-2.5"]
    if any(pro_model in original_model for pro_model in pro_models):
        location = gemini2
        search_tool = {"googleSearch": {}}
    else:
        location = gemini1
        search_tool = {"googleSearchRetrieval": {}}

    if "google-vertex-ai" in provider.get("base_url", ""):
        url = provider.get("base_url").rstrip('/') + "/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(
            LOCATION=await location.next(),
            PROJECT_ID=project_id,
            MODEL_ID=original_model,
            stream=gemini_stream
        )
    else:
        url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(
            LOCATION=await location.next(),
            PROJECT_ID=project_id,
            MODEL_ID=original_model,
            stream=gemini_stream
        )

    messages = []
    systemInstruction = None
    function_arguments = None
    for msg in request.messages:
        if msg.role == "assistant":
            msg.role = "model"
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = [{"text": msg.content}]
            tool_calls = msg.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_arguments = {
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
            }
            messages.append(
                {
                    "role": "model",
                    "parts": [function_arguments]
                }
            )
        elif msg.role == "tool":
            function_call_name = function_arguments["functionCall"]["name"]
            messages.append(
                {
                    "role": "function",
                    "parts": [{
                    "functionResponse": {
                        "name": function_call_name,
                        "response": {
                            "name": function_call_name,
                            "content": {
                                "result": msg.content,
                            }
                        }
                    }
                    }]
                }
            )
        elif msg.role != "system":
            messages.append({"role": msg.role, "parts": content})
        elif msg.role == "system":
            systemInstruction = {"parts": content}


    payload = {
        "contents": messages,
        # "safetySettings": [
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_NONE"
        #     }
        # ]
    }
    if systemInstruction:
        payload["system_instruction"] = systemInstruction

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
    ]
    generation_config = {}

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools":
                payload.update({
                    "tools": [{
                        "function_declarations": [tool["function"] for tool in value]
                    }],
                    "tool_config": {
                        "function_calling_config": {
                            "mode": "AUTO"
                        }
                    }
                })
            elif field == "temperature":
                generation_config["temperature"] = value
            elif field == "max_tokens":
                generation_config["max_output_tokens"] = value
            elif field == "top_p":
                generation_config["top_p"] = value
            else:
                payload[field] = value

    if generation_config:
        payload["generationConfig"] = generation_config
        if "max_output_tokens" not in generation_config:
            payload["generationConfig"]["max_output_tokens"] = 8192

    if request.model.endswith("-search"):
        if "tools" not in payload:
            payload["tools"] = [search_tool]
        else:
            payload["tools"].append(search_tool)

    return url, headers, payload

async def get_vertex_claude_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if "claude-3-5-sonnet" in original_model or "claude-3-7-sonnet" in original_model:
        location = c35s
    elif "claude-3-opus" in original_model:
        location = c3o
    elif "claude-3-sonnet" in original_model:
        location = c3s
    elif "claude-3-haiku" in original_model:
        location = c3h

    claude_stream = "streamRawPredict"
    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:{stream}".format(
        LOCATION=await location.next(),
        PROJECT_ID=project_id,
        MODEL=original_model,
        stream=claude_stream
    )

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    if "claude-3-7-sonnet" in original_model:
        max_tokens = 20000
    elif "claude-3-5-sonnet" in original_model:
        max_tokens = 8192
    else:
        max_tokens = 4096

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    return url, headers, payload

import hashlib
import hmac
import datetime
import urllib.parse
from datetime import timezone

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing

def get_signature(request_body, model_id, aws_access_key, aws_secret_key, aws_region, host, content_type, accept_header):
    request_body = json.dumps(request_body)
    SERVICE = "bedrock"
    canonical_querystring = ''
    method = 'POST'
    raw_path = f'/model/{model_id}/invoke-with-response-stream'
    canonical_uri = urllib.parse.quote(raw_path, safe='/-_.~')
    # Create a date for headers and the credential string
    t = datetime.datetime.now(timezone.utc)
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d') # Date YYYYMMDD

    # --- Task 1: Create a Canonical Request ---
    payload_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()

    canonical_headers = f'accept:{accept_header}\n' \
                        f'content-type:{content_type}\n' \
                        f'host:{host}\n' \
                        f'x-amz-bedrock-accept:{accept_header}\n' \
                        f'x-amz-content-sha256:{payload_hash}\n' \
                        f'x-amz-date:{amz_date}\n'
    # 注意：头名称需要按字母顺序排序

    signed_headers = 'accept;content-type;host;x-amz-bedrock-accept;x-amz-content-sha256;x-amz-date' # 按字母顺序排序

    canonical_request = f'{method}\n' \
                        f'{canonical_uri}\n' \
                        f'{canonical_querystring}\n' \
                        f'{canonical_headers}\n' \
                        f'{signed_headers}\n' \
                        f'{payload_hash}'

    # --- Task 2: Create the String to Sign ---
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date_stamp}/{aws_region}/{SERVICE}/aws4_request'
    string_to_sign = f'{algorithm}\n' \
                    f'{amz_date}\n' \
                    f'{credential_scope}\n' \
                    f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'

    # --- Task 3: Calculate the Signature ---
    signing_key = get_signature_key(aws_secret_key, date_stamp, aws_region, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    # --- Task 4: Add Signing Information to the Request ---
    authorization_header = f'{algorithm} Credential={aws_access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}'
    return amz_date, payload_hash, authorization_header

async def get_aws_payload(request, engine, provider, api_key=None):
    CONTENT_TYPE = "application/json"
    # AWS_REGION = "us-east-1"
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    # MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    base_url = provider.get('base_url')
    AWS_REGION = base_url.split('.')[1]
    HOST = f"bedrock-runtime.{AWS_REGION}.amazonaws.com"
    # url = f"{base_url}/model/{original_model}/invoke"
    url = f"{base_url}/model/{original_model}/invoke-with-response-stream"

    # if "claude-3-5-sonnet" in original_model or "claude-3-7-sonnet" in original_model:
    #     location = c35s
    # elif "claude-3-opus" in original_model:
    #     location = c3o
    # elif "claude-3-sonnet" in original_model:
    #     location = c3s
    # elif "claude-3-haiku" in original_model:
    #     location = c3h

    # claude_stream = "streamRawPredict"
    # url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:{stream}".format(
    #     LOCATION=await location.next(),
    #     PROJECT_ID=project_id,
    #     MODEL=original_model,
    #     stream=claude_stream
    # )

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    # if "claude-3-7-sonnet" in original_model:
    #     max_tokens = 20000
    # elif "claude-3-5-sonnet" in original_model:
    #     max_tokens = 8192
    # else:
    #     max_tokens = 4096
    max_tokens = 4096

    payload = {
        "messages": messages,
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
    }

    # payload = {
    #     "anthropic_version": "vertex-2023-10-16",
    #     "messages": messages,
    #     "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
    #     "max_tokens": max_tokens,
    # }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
        'stream',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if provider.get("aws_access_key") and provider.get("aws_secret_key"):
        ACCEPT_HEADER = "application/vnd.amazon.bedrock.payload+json" # 指定接受 Bedrock 流格式
        amz_date, payload_hash, authorization_header = get_signature(payload, original_model, provider.get("aws_access_key"), provider.get("aws_secret_key"), AWS_REGION, HOST, CONTENT_TYPE, ACCEPT_HEADER)
        headers = {
            'Accept': ACCEPT_HEADER,
            'Content-Type': CONTENT_TYPE,
            'X-Amz-Date': amz_date,
            'X-Amz-Bedrock-Accept': ACCEPT_HEADER, # Bedrock 特定头
            'X-Amz-Content-Sha256': payload_hash,
            'Authorization': authorization_header,
            # Add 'X-Amz-Security-Token': SESSION_TOKEN if using temporary credentials
        }

    return url, headers, payload

async def get_gpt_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True) and "o1-mini" not in original_model:
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            if msg.role == "system" and "o3-mini" in original_model and not content.startswith("Formatting re-enabled"):
                content = "Formatting re-enabled. " + content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            messages.append({"role": msg.role, "content": content})

    if ("o1-mini" in original_model or "o1-preview" in original_model) and len(messages) > 1 and messages[0]["role"] == "system":
        system_msg = messages.pop(0)
        messages[0]["content"] = system_msg["content"] + messages[0]["content"]

    payload = {
        "model": original_model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and ("o1" in original_model or "o3" in original_model or "o4" in original_model):
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") == False or "o1-mini" in original_model or "chatgpt-4o-latest" in original_model or "grok" in original_model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "api.x.ai" in url:
        payload.pop("stream_options", None)
        payload.pop("presence_penalty", None)
        payload.pop("frequency_penalty", None)

    if "grok-3-mini" in original_model:
        if request.model.endswith("high"):
            payload["reasoning_effort"] = "high"
        elif request.model.endswith("low"):
            payload["reasoning_effort"] = "low"

    if "o1" in original_model or "o3" in original_model or "o4" in original_model:
        if request.model.endswith("high"):
            payload["reasoning_effort"] = "high"
        elif request.model.endswith("low"):
            payload["reasoning_effort"] = "low"
        else:
            payload["reasoning_effort"] = "medium"

        if "temperature" in payload:
            payload.pop("temperature")

    # 代码生成/数学解题  0.0
    # 数据抽取/分析	     1.0
    # 通用对话          1.3
    # 翻译	           1.3
    # 创意类写作/诗歌创作 1.5
    if "deepseek-r" in original_model.lower():
        if "temperature" not in payload:
            payload["temperature"] = 0.6

    if request.model.endswith("-search") and "gemini" in original_model:
        if "tools" not in payload:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "googleSearch",
                    "description": "googleSearch"
                }
            }]
        else:
            if not any(tool["function"]["name"] == "googleSearch" for tool in payload["tools"]):
                payload["tools"].append({
                    "type": "function",
                    "function": {
                        "name": "googleSearch",
                        "description": "googleSearch"
                    }
                })

    if safe_get(provider, "preferences", "post_body_parameter_overrides", default=None):
        for key, value in safe_get(provider, "preferences", "post_body_parameter_overrides", default={}).items():
            payload[key] = value

    return url, headers, payload

def build_azure_endpoint(base_url, deployment_id, api_version="2025-01-01-preview"):
    # 移除base_url末尾的斜杠(如果有)
    base_url = base_url.rstrip('/')
    final_url = base_url

    if "models/chat/completions" not in final_url:
        # 构建路径
        path = f"/openai/deployments/{deployment_id}/chat/completions"
        # 使用urljoin拼接base_url和path
        final_url = urllib.parse.urljoin(base_url, path)

    if "?api-version=" not in final_url:
        # 添加api-version查询参数
        final_url = f"{final_url}?api-version={api_version}"

    return final_url

async def get_azure_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers['api-key'] = f"{api_key}"

    url = build_azure_endpoint(
        base_url=provider['base_url'],
        deployment_id=original_model,
    )

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True) and "o1-mini" not in original_model:
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            messages.append({"role": msg.role, "content": content})

    payload = {
        "model": original_model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and "o1" in original_model:
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") == False or "o1" in original_model or "chatgpt-4o-latest" in original_model or "grok" in original_model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if safe_get(provider, "preferences", "post_body_parameter_overrides", default=None):
        for key, value in safe_get(provider, "preferences", "post_body_parameter_overrides", default={}).items():
            payload[key] = value

    return url, headers, payload

async def get_openrouter_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']

    messages = []
    for msg in request.messages:
        name = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            name = msg.name
        if name:
            messages.append({"role": msg.role, "name": name, "content": content})
        else:
            # print("content", content)
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        messages.append({"role": msg.role, "content": item["text"]})
                    elif item["type"] == "image_url":
                        messages.append({"role": msg.role, "content": [await get_image_message(item["image_url"]["url"], engine)]})
            else:
                messages.append({"role": msg.role, "content": content})

    payload = {
        "model": original_model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def get_cohere_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']

    role_map = {
        "user": "USER",
        "assistant" : "CHATBOT",
        "system": "SYSTEM"
    }

    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
        else:
            content = msg.content

        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    messages.append({"role": role_map[msg.role], "message": item["text"]})
        else:
            messages.append({"role": role_map[msg.role], "message": content})

    chat_history = messages[:-1]
    query = messages[-1].get("message")
    payload = {
        "model": original_model,
        "message": query,
    }

    if chat_history:
        payload["chat_history"] = chat_history

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def get_cloudflare_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = "https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{cf_model_id}".format(cf_account_id=provider['cf_account_id'], cf_model_id=original_model)

    msg = request.messages[-1]
    content = None
    if isinstance(msg.content, list):
        for item in msg.content:
            if item.type == "text":
                content = await get_text_message(item.text, engine)
    else:
        content = msg.content

    payload = {
        "prompt": content,
    }

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def gpt2claude_tools_json(json_dict):
    import copy
    json_dict = copy.deepcopy(json_dict)

    # 处理 $ref 引用
    def resolve_refs(obj, defs):
        if isinstance(obj, dict):
            # 如果有 $ref 引用，替换为实际定义
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    # 完全替换为引用的对象
                    ref_obj = copy.deepcopy(defs[ref_name])
                    # 保留原始对象中的其他属性
                    for k, v in obj.items():
                        if k != "$ref":
                            ref_obj[k] = v
                    return ref_obj

            # 递归处理所有属性
            for key, value in list(obj.items()):
                obj[key] = resolve_refs(value, defs)

        elif isinstance(obj, list):
            # 递归处理列表中的每个元素
            for i, item in enumerate(obj):
                obj[i] = resolve_refs(item, defs)

        return obj

    # 提取 $defs 定义
    defs = {}
    if "parameters" in json_dict and "defs" in json_dict["parameters"]:
        defs = json_dict["parameters"]["defs"]
        # 从参数中删除 $defs，因为 Claude 不需要它
        del json_dict["parameters"]["defs"]

    # 解析所有引用
    json_dict = resolve_refs(json_dict, defs)

    # 继续原有的键名转换逻辑
    keys_to_change = {
        "parameters": "input_schema",
    }
    for old_key, new_key in keys_to_change.items():
        if old_key in json_dict:
            if new_key:
                if json_dict[old_key] == None:
                    json_dict[old_key] = {
                        "type": "object",
                        "properties": {}
                    }
                json_dict[new_key] = json_dict.pop(old_key)
            else:
                json_dict.pop(old_key)
    return json_dict

async def get_claude_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    if "claude-3-7-sonnet" in original_model:
        anthropic_beta = "output-128k-2025-02-19"
    elif "claude-3-5-sonnet" in original_model:
        anthropic_beta = "max-tokens-3-5-sonnet-2024-07-15"
    else:
        anthropic_beta = "tools-2024-05-16"

    headers = {
        "content-type": "application/json",
        "x-api-key": f"{api_key}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": anthropic_beta,
    }
    url = provider['base_url']

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            messages.append({"role": msg.role, "content": content})
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    if "claude-3-7-sonnet" in original_model:
        max_tokens = 20000
    elif "claude-3-5-sonnet" in original_model:
        max_tokens = 8192
    else:
        max_tokens = 4096

    payload = {
        "model": original_model,
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            # print("tool", type(tool), tool)
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") == False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "think" in request.model:
        payload["thinking"] = {
            "budget_tokens": 4096,
            "type": "enabled"
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        if request.model.split("-")[-1].isdigit():
            think_tokens = int(request.model.split("-")[-1])
            if think_tokens < max_tokens:
                payload["thinking"] = {
                    "budget_tokens": think_tokens,
                    "type": "enabled"
                }

    if request.thinking:
        payload["thinking"] = {
            "budget_tokens": request.thinking.budget_tokens,
            "type": request.thinking.type
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
    # print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    return url, headers, payload

async def get_dalle_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).image_url

    payload = {
        "model": original_model,
        "prompt": request.prompt,
        "n": request.n,
        "response_format": request.response_format,
        "size": request.size
    }

    return url, headers, payload

async def get_whisper_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        # "Content-Type": "multipart/form-data",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).audio_transcriptions

    payload = {
        "model": original_model,
        "file": request.file,
    }

    if request.prompt:
        payload["prompt"] = request.prompt
    if request.response_format:
        payload["response_format"] = request.response_format
    if request.temperature:
        payload["temperature"] = request.temperature
    if request.language:
        payload["language"] = request.language

    # https://platform.openai.com/docs/api-reference/audio/createTranscription
    if request.timestamp_granularities:
        payload["timestamp_granularities[]"] = request.timestamp_granularities

    return url, headers, payload

async def get_moderation_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).moderations

    payload = {
        "model": original_model,
        "input": request.input,
    }

    return url, headers, payload

async def get_embedding_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).embeddings

    payload = {
        "input": request.input,
        "model": original_model,
    }

    if request.encoding_format:
        if url.startswith("https://api.jina.ai"):
            payload["embedding_type"] = request.encoding_format
        else:
            payload["encoding_format"] = request.encoding_format

    return url, headers, payload

async def get_tts_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).audio_speech

    payload = {
        "model": original_model,
        "input": request.input,
        "voice": request.voice,
    }

    if request.response_format:
        payload["response_format"] = request.response_format
    if request.speed:
        payload["speed"] = request.speed
    if request.stream is not None:
        payload["stream"] = request.stream

    return url, headers, payload


async def get_payload(request: RequestModel, engine, provider, api_key=None):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider, api_key)
    elif engine == "vertex-gemini":
        return await get_vertex_gemini_payload(request, engine, provider, api_key)
    elif engine == "aws":
        return await get_aws_payload(request, engine, provider, api_key)
    elif engine == "vertex-claude":
        return await get_vertex_claude_payload(request, engine, provider, api_key)
    elif engine == "azure":
        return await get_azure_payload(request, engine, provider, api_key)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider, api_key)
    elif engine == "gpt":
        provider['base_url'] = BaseAPI(provider['base_url']).chat_url
        return await get_gpt_payload(request, engine, provider, api_key)
    elif engine == "openrouter":
        return await get_openrouter_payload(request, engine, provider, api_key)
    elif engine == "cloudflare":
        return await get_cloudflare_payload(request, engine, provider, api_key)
    elif engine == "cohere":
        return await get_cohere_payload(request, engine, provider, api_key)
    elif engine == "dalle":
        return await get_dalle_payload(request, engine, provider, api_key)
    elif engine == "whisper":
        return await get_whisper_payload(request, engine, provider, api_key)
    elif engine == "tts":
        return await get_tts_payload(request, engine, provider, api_key)
    elif engine == "moderation":
        return await get_moderation_payload(request, engine, provider, api_key)
    elif engine == "embedding":
        return await get_embedding_payload(request, engine, provider, api_key)
    else:
        raise ValueError("Unknown payload")

async def prepare_request_payload(provider, request_data):

    model_dict = get_model_dict(provider)
    request = RequestModel(**request_data)

    original_model = model_dict[request.model]
    engine, _ = get_engine(provider, endpoint=None, original_model=original_model)

    url, headers, payload = await get_payload(request, engine, provider, api_key=provider['api'])

    return url, headers, payload, engine