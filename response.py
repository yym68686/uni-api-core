import re
import json
import random
import string
import base64
import asyncio
from datetime import datetime

from .log_config import logger

from .utils import safe_get, generate_sse_response, generate_no_stream_response, end_of_line, parse_json_safely

async def check_response(response, error_log):
    if response and not (200 <= response.status_code < 300):
        error_message = await response.aread()
        error_str = error_message.decode('utf-8', errors='replace')
        try:
            error_json = await asyncio.to_thread(json.loads, error_str)
        except json.JSONDecodeError:
            error_json = error_str
        return {"error": f"{error_log} HTTP Error", "status_code": response.status_code, "details": error_json}
    return None

async def gemini_json_poccess(response_str):
    promptTokenCount = 0
    candidatesTokenCount = 0
    totalTokenCount = 0
    image_base64 = None

    response_json = await asyncio.to_thread(json.loads, response_str)
    json_data = safe_get(response_json, "candidates", 0, "content", default=None)
    finishReason = safe_get(response_json, "candidates", 0 , "finishReason", default=None)
    if finishReason:
        promptTokenCount = safe_get(response_json, "usageMetadata", "promptTokenCount", default=0)
        candidatesTokenCount = safe_get(response_json, "usageMetadata", "candidatesTokenCount", default=0)
        totalTokenCount = safe_get(response_json, "usageMetadata", "totalTokenCount", default=0)
        if finishReason != "STOP":
            logger.error(f"finishReason: {finishReason}")

    content = reasoning_content = safe_get(json_data, "parts", 0, "text", default="")
    b64_json = safe_get(json_data, "parts", 0, "inlineData", "data", default="")
    if b64_json:
        image_base64 = b64_json

    is_thinking = safe_get(json_data, "parts", 0, "thought", default=False)
    if is_thinking:
        content = safe_get(json_data, "parts", 1, "text", default="")

    function_call_name = safe_get(json_data, "functionCall", "name", default=None)
    function_full_response = safe_get(json_data, "functionCall", "args", default="")
    function_full_response = json.dumps(function_full_response) if function_full_response else None

    blockReason = safe_get(json_data, 0, "promptFeedback", "blockReason", default=None)

    return is_thinking, reasoning_content, content, image_base64, function_call_name, function_full_response, finishReason, blockReason, promptTokenCount, candidatesTokenCount, totalTokenCount

async def fetch_gemini_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_gemini_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0
        parts_json = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            if buffer and "\n" not in buffer:
                buffer += "\n"

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.startswith("data: "):
                    parts_json = line.lstrip("data: ").strip()
                    try:
                        await asyncio.to_thread(json.loads, parts_json)
                    except json.JSONDecodeError:
                        logger.error(f"JSON decode error: {parts_json}")
                        continue
                else:
                    parts_json += line
                    parts_json = parts_json.lstrip("[,")
                    try:
                        await asyncio.to_thread(json.loads, parts_json)
                    except json.JSONDecodeError:
                        continue

                # https://ai.google.dev/api/generate-content?hl=zh-cn#FinishReason
                is_thinking, reasoning_content, content, image_base64, function_call_name, function_full_response, finishReason, blockReason, promptTokenCount, candidatesTokenCount, totalTokenCount = await gemini_json_poccess(parts_json)

                if is_thinking:
                    sse_string = await generate_sse_response(timestamp, model, reasoning_content=reasoning_content)
                    yield sse_string
                if not image_base64 and content:
                    sse_string = await generate_sse_response(timestamp, model, content=content)
                    yield sse_string

                if image_base64:
                    yield await generate_no_stream_response(timestamp, model, content=content, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=totalTokenCount, prompt_tokens=promptTokenCount, completion_tokens=candidatesTokenCount, image_base64=image_base64)

                if function_call_name:
                    sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=function_call_name)
                    yield sse_string
                if function_full_response:
                    sse_string = await generate_sse_response(timestamp, model, content=None, tools_id="chatcmpl-9inWv0yEtgn873CxMBzHeCeiHctTV", function_call_name=None, function_call_content=function_full_response)
                    yield sse_string

                if parts_json == "[]" or blockReason == "PROHIBITED_CONTENT":
                    sse_string = await generate_sse_response(timestamp, model, stop="PROHIBITED_CONTENT")
                    yield sse_string
                elif finishReason:
                    sse_string = await generate_sse_response(timestamp, model, stop="stop")
                    yield sse_string
                    break

                parts_json = ""

        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, totalTokenCount, promptTokenCount, candidatesTokenCount)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_vertex_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_vertex_claude_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        revicing_function_call = False
        function_full_response = "{"
        need_function_call = False
        is_finish = False
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0

        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(f"{line}")

                if line and '\"finishReason\": \"' in line:
                    is_finish = True
                if is_finish and '\"promptTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    promptTokenCount = json_data.get('promptTokenCount', 0)
                if is_finish and '\"candidatesTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    candidatesTokenCount = json_data.get('candidatesTokenCount', 0)
                if is_finish and '\"totalTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    totalTokenCount = json_data.get('totalTokenCount', 0)

                if line and '\"text\": \"' in line and is_finish == False:
                    try:
                        json_data = await asyncio.to_thread(json.loads, "{" + line.strip().rstrip(",") + "}")
                        content = json_data.get('text', '')
                        sse_string = await generate_sse_response(timestamp, model, content=content)
                        yield sse_string
                    except json.JSONDecodeError:
                        logger.error(f"无法解析JSON: {line}")

                if line and ('\"type\": \"tool_use\"' in line or revicing_function_call):
                    revicing_function_call = True
                    need_function_call = True
                    if ']' in line:
                        revicing_function_call = False
                        continue

                    function_full_response += line

        if need_function_call:
            function_call = await asyncio.to_thread(json.loads, function_full_response)
            function_call_name = function_call["name"]
            function_call_id = function_call["id"]
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=function_call_name)
            yield sse_string
            function_full_response = json.dumps(function_call["input"])
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=None, function_call_content=function_full_response)
            yield sse_string

        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, totalTokenCount, promptTokenCount, candidatesTokenCount)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_gpt_response_stream(client, url, headers, payload, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_gpt_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        enter_buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and not line.startswith(":") and (result:=line.lstrip("data: ").strip()):
                    if result.strip() == "[DONE]":
                        break
                    line = await asyncio.to_thread(json.loads, result)
                    line['id'] = f"chatcmpl-{random_str}"

                    # 处理 <think> 标签
                    content = safe_get(line, "choices", 0, "delta", "content", default="")
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        end_think_reasoning_content = ""
                        end_think_content = ""
                        is_thinking = False

                        if content.rstrip('\n').endswith("</think>"):
                            end_think_reasoning_content = content.replace("</think>", "").rstrip('\n')
                        elif content.lstrip('\n').startswith("</think>"):
                            end_think_content = content.replace("</think>", "").lstrip('\n')
                        else:
                            end_think_reasoning_content = content.split("</think>")[0]
                            end_think_content = content.split("</think>")[1]

                        if end_think_reasoning_content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=end_think_reasoning_content)
                            yield sse_string
                        if end_think_content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], content=end_think_content)
                            yield sse_string
                        continue
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    # 处理 poe thinking 标签
                    if "Thinking..." in content and "\n> " in content:
                        is_thinking = True
                        content = content.replace("Thinking...", "").replace("\n> ", "")
                    if is_thinking and "\n\n" in content and not ark_tag:
                        is_thinking = False
                    if is_thinking and not ark_tag:
                        content = content.replace("\n> ", "")
                        if not has_send_thinking:
                            content = content.replace("\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    no_stream_content = safe_get(line, "choices", 0, "message", "content", default=None)
                    openrouter_reasoning = safe_get(line, "choices", 0, "delta", "reasoning", default="")
                    azure_databricks_claude_summary_content = safe_get(line, "choices", 0, "delta", "content", 0, "summary", 0, "text", default="")
                    azure_databricks_claude_signature_content = safe_get(line, "choices", 0, "delta", "content", 0, "summary", 0, "signature", default="")
                    # print("openrouter_reasoning", repr(openrouter_reasoning), openrouter_reasoning.endswith("\\\\"), openrouter_reasoning.endswith("\\"))
                    if azure_databricks_claude_signature_content:
                        pass
                    elif azure_databricks_claude_summary_content:
                        sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=azure_databricks_claude_summary_content)
                        yield sse_string
                    elif openrouter_reasoning:
                        if openrouter_reasoning.endswith("\\"):
                            enter_buffer += openrouter_reasoning
                            continue
                        elif enter_buffer.endswith("\\") and openrouter_reasoning == 'n':
                            enter_buffer += "n"
                            continue
                        elif enter_buffer.endswith("\\n") and openrouter_reasoning == '\\n':
                            enter_buffer += "\\n"
                            continue
                        elif enter_buffer.endswith("\\n\\n"):
                            openrouter_reasoning = '\n\n' + openrouter_reasoning
                            enter_buffer = ""
                        elif enter_buffer:
                            openrouter_reasoning = enter_buffer + openrouter_reasoning
                            enter_buffer = ''
                        openrouter_reasoning = openrouter_reasoning.replace("\\n", "\n")

                        sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=openrouter_reasoning)
                        yield sse_string
                    elif no_stream_content and has_send_thinking == False:
                        sse_string = await generate_sse_response(safe_get(line, "created", default=None), safe_get(line, "model", default=None), content=no_stream_content)
                        yield sse_string
                    else:
                        if no_stream_content:
                            del line["choices"][0]["message"]
                        yield "data: " + json.dumps(line).strip() + end_of_line
    yield "data: [DONE]" + end_of_line

async def fetch_azure_response_stream(client, url, headers, payload, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_azure_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        sse_string = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and not line.startswith(":") and (result:=line.lstrip("data: ").strip()):
                    if result.strip() == "[DONE]":
                        break
                    line = await asyncio.to_thread(json.loads, result)
                    no_stream_content = safe_get(line, "choices", 0, "message", "content", default="")
                    content = safe_get(line, "choices", 0, "delta", "content", default="")

                    # 处理 <think> 标签
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        is_thinking = False
                        content = content.replace("</think>", "")
                        if not content:
                            continue
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    if no_stream_content or content or sse_string:
                        input_tokens = safe_get(line, "usage", "prompt_tokens", default=0)
                        output_tokens = safe_get(line, "usage", "completion_tokens", default=0)
                        total_tokens = safe_get(line, "usage", "total_tokens", default=0)
                        sse_string = await generate_sse_response(timestamp, safe_get(line, "model", default=None), content=no_stream_content or content, total_tokens=total_tokens, prompt_tokens=input_tokens, completion_tokens=output_tokens)
                        yield sse_string
                    else:
                        if no_stream_content:
                            del line["choices"][0]["message"]
                        yield "data: " + json.dumps(line).strip() + end_of_line
    yield "data: [DONE]" + end_of_line

async def fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cloudflare_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line.startswith("data:"):
                    line = line.lstrip("data: ")
                    if line == "[DONE]":
                        break
                    resp: dict = await asyncio.to_thread(json.loads, line)
                    message = resp.get("response")
                    if message:
                        sse_string = await generate_sse_response(timestamp, model, content=message)
                        yield sse_string
    yield "data: [DONE]" + end_of_line

async def fetch_cohere_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cohere_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                resp: dict = await asyncio.to_thread(json.loads, line)
                if resp.get("is_finished") == True:
                    break
                if resp.get("event_type") == "text-generation":
                    message = resp.get("text")
                    sse_string = await generate_sse_response(timestamp, model, content=message)
                    yield sse_string
    yield "data: [DONE]" + end_of_line

async def fetch_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_claude_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        input_tokens = 0
        async for chunk in response.aiter_text():
            # logger.info(f"chunk: {repr(chunk)}")
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(line)

                if line.startswith("data:") and (line := line.lstrip("data: ")):
                    resp: dict = await asyncio.to_thread(json.loads, line)

                    input_tokens = input_tokens or safe_get(resp, "message", "usage", "input_tokens", default=0)
                    # cache_creation_input_tokens = safe_get(resp, "message", "usage", "cache_creation_input_tokens", default=0)
                    # cache_read_input_tokens = safe_get(resp, "message", "usage", "cache_read_input_tokens", default=0)
                    output_tokens = safe_get(resp, "usage", "output_tokens", default=0)
                    if output_tokens:
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, total_tokens, input_tokens, output_tokens)
                        yield sse_string
                        break

                    text = safe_get(resp, "delta", "text", default="")
                    if text:
                        sse_string = await generate_sse_response(timestamp, model, text)
                        yield sse_string
                        continue

                    function_call_name = safe_get(resp, "content_block", "name", default=None)
                    tools_id = safe_get(resp, "content_block", "id", default=None)
                    if tools_id and function_call_name:
                        sse_string = await generate_sse_response(timestamp, model, None, tools_id, function_call_name, None)
                        yield sse_string

                    thinking_content = safe_get(resp, "delta", "thinking", default="")
                    if thinking_content:
                        sse_string = await generate_sse_response(timestamp, model, reasoning_content=thinking_content)
                        yield sse_string

                    function_call_content = safe_get(resp, "delta", "partial_json", default="")
                    if function_call_content:
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, function_call_content)
                        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_aws_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    async with client.stream('POST', url, headers=headers, json=payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_aws_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for line in response.aiter_text():
            buffer += line
            while "\r" in buffer:
                line, buffer = buffer.split("\r", 1)
                if not line or \
                line.strip() == "" or \
                line.strip().startswith(':content-type') or \
                line.strip().startswith(':event-type'): # 过滤掉完全空的行或只有空白的行
                    continue

                json_match = re.search(r'event{.*?}', line)
                if not json_match:
                    continue
                try:
                    chunk_data = await asyncio.to_thread(json.loads, json_match.group(0).lstrip('event'))
                except json.JSONDecodeError:
                    logger.error(f"DEBUG json.JSONDecodeError: {json_match.group(0).lstrip('event')!r}")
                    continue

                # --- 后续处理逻辑不变 ---
                if "bytes" in chunk_data:
                    # 解码 Base64 编码的字节
                    decoded_bytes = base64.b64decode(chunk_data["bytes"])
                    # 将解码后的字节再次解析为 JSON
                    payload_chunk = await asyncio.to_thread(json.loads, decoded_bytes.decode('utf-8'))
                    # print(f"DEBUG payload_chunk: {payload_chunk!r}")

                    text = safe_get(payload_chunk, "delta", "text", default="")
                    if text:
                        sse_string = await generate_sse_response(timestamp, model, text, None, None)
                        yield sse_string

                    usage = safe_get(payload_chunk, "amazon-bedrock-invocationMetrics", default="")
                    if usage:
                        input_tokens = usage.get("inputTokenCount", 0)
                        output_tokens = usage.get("outputTokenCount", 0)
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, total_tokens, input_tokens, output_tokens)
                        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_response(client, url, headers, payload, engine, model, timeout=200):
    response = None
    if payload.get("file"):
        file = payload.pop("file")
        response = await client.post(url, headers=headers, data=payload, files={"file": file}, timeout=timeout)
    else:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
    error_message = await check_response(response, "fetch_response")
    if error_message:
        yield error_message
        return

    if engine == "tts":
        yield response.read()

    elif engine == "gemini" or engine == "vertex-gemini" or engine == "aws":
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        # print("response_json", json.dumps(response_json, indent=4, ensure_ascii=False))

        if isinstance(response_json, str):
            import ast
            parsed_data = ast.literal_eval(str(response_json))
        elif isinstance(response_json, list):
            parsed_data = response_json
        elif isinstance(response_json, dict):
            parsed_data = [response_json]
        else:
            logger.error(f"error fetch_response: Unknown response_json type: {type(response_json)}")
            parsed_data = response_json

        content = ""
        reasoning_content = ""
        image_base64 = ""
        parts_list = safe_get(parsed_data, 0, "candidates", 0, "content", "parts", default=[])
        for item in parts_list:
            chunk = safe_get(item, "text")
            b64_json = safe_get(item, "inlineData", "data", default="")
            if b64_json:
                image_base64 = b64_json
            is_think = safe_get(item, "thought", default=False)
            # logger.info(f"chunk: {repr(chunk)}")
            if chunk:
                if is_think:
                    reasoning_content += chunk
                else:
                    content += chunk

        usage_metadata = safe_get(parsed_data, -1, "usageMetadata")
        prompt_tokens = safe_get(usage_metadata, "promptTokenCount", default=0)
        candidates_tokens = safe_get(usage_metadata, "candidatesTokenCount", default=0)
        total_tokens = safe_get(usage_metadata, "totalTokenCount", default=0)

        role = safe_get(parsed_data, -1, "candidates", 0, "content", "role")
        if role == "model":
            role = "assistant"
        else:
            logger.error(f"Unknown role: {role}, parsed_data: {parsed_data}")
            role = "assistant"

        has_think = safe_get(parsed_data, 0, "candidates", 0, "content", "parts", 0, "thought", default=False)
        if has_think:
            function_message_parts_index = -1
        else:
            function_message_parts_index = 0
        function_call_name = safe_get(parsed_data, -1, "candidates", 0, "content", "parts", function_message_parts_index, "functionCall", "name", default=None)
        function_call_content = safe_get(parsed_data, -1, "candidates", 0, "content", "parts", function_message_parts_index, "functionCall", "args", default=None)

        timestamp = int(datetime.timestamp(datetime.now()))
        yield await generate_no_stream_response(timestamp, model, content=content, tools_id=None, function_call_name=function_call_name, function_call_content=function_call_content, role=role, total_tokens=total_tokens, prompt_tokens=prompt_tokens, completion_tokens=candidates_tokens, reasoning_content=reasoning_content, image_base64=image_base64)

    elif engine == "claude":
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        # print("response_json", json.dumps(response_json, indent=4, ensure_ascii=False))

        content = safe_get(response_json, "content", 0, "text")

        prompt_tokens = safe_get(response_json, "usage", "input_tokens")
        output_tokens = safe_get(response_json, "usage", "output_tokens")
        total_tokens = prompt_tokens + output_tokens

        role = safe_get(response_json, "role")

        function_call_name = safe_get(response_json, "content", 1, "name", default=None)
        function_call_content = safe_get(response_json, "content", 1, "input", default=None)
        tools_id = safe_get(response_json, "content", 1, "id", default=None)

        timestamp = int(datetime.timestamp(datetime.now()))
        yield await generate_no_stream_response(timestamp, model, content=content, tools_id=tools_id, function_call_name=function_call_name, function_call_content=function_call_content, role=role, total_tokens=total_tokens, prompt_tokens=prompt_tokens, completion_tokens=output_tokens)

    elif engine == "azure":
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        # 删除 content_filter_results
        if "choices" in response_json:
            for choice in response_json["choices"]:
                if "content_filter_results" in choice:
                    del choice["content_filter_results"]

        # 删除 prompt_filter_results
        if "prompt_filter_results" in response_json:
            del response_json["prompt_filter_results"]

        yield response_json

    elif "dashscope.aliyuncs.com" in url and "multimodal-generation" in url:
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        content = safe_get(response_json, "output", "choices", 0, "message", "content", 0, default=None)
        yield content

    elif "embedContent" in url:
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        content = safe_get(response_json, "embedding", "values", default=[])
        response_embedContent = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding":content,
                    "index": 0
                }
            ],
            "model": model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }

        yield response_embedContent
    else:
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        yield response_json

async def fetch_response_stream(client, url, headers, payload, engine, model, timeout=200):
    if engine == "gemini" or engine == "vertex-gemini":
        async for chunk in fetch_gemini_response_stream(client, url, headers, payload, model, timeout):
            yield chunk
    elif engine == "claude" or engine == "vertex-claude":
        async for chunk in fetch_claude_response_stream(client, url, headers, payload, model, timeout):
            yield chunk
    elif engine == "aws":
        async for chunk in fetch_aws_response_stream(client, url, headers, payload, model, timeout):
            yield chunk
    elif engine == "gpt" or engine == "openrouter" or engine == "azure-databricks":
        async for chunk in fetch_gpt_response_stream(client, url, headers, payload, timeout):
            yield chunk
    elif engine == "azure":
        async for chunk in fetch_azure_response_stream(client, url, headers, payload, timeout):
            yield chunk
    elif engine == "cloudflare":
        async for chunk in fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout):
            yield chunk
    elif engine == "cohere":
        async for chunk in fetch_cohere_response_stream(client, url, headers, payload, model, timeout):
            yield chunk
    else:
        raise ValueError("Unknown response")
