import re
import asyncio
from time import time
from fastapi import HTTPException
from urllib.parse import urlparse
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("uni-api-core")

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("watchfiles.main").setLevel(logging.CRITICAL)

def get_model_dict(provider):
    model_dict = {}
    for model in provider['model']:
        if type(model) == str:
            model_dict[model] = model
        if isinstance(model, dict):
            model_dict.update({new: old for old, new in model.items()})
    return model_dict

class BaseAPI:
    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        if api_url == "":
            api_url = "https://api.openai.com/v1/chat/completions"
        self.source_api_url: str = api_url
        from urllib.parse import urlparse, urlunparse
        parsed_url = urlparse(self.source_api_url)
        # print("parsed_url", parsed_url)
        if parsed_url.scheme == "":
            raise Exception("Error: API_URL is not set")
        if parsed_url.path != '/':
            before_v1 = parsed_url.path.split("chat/completions")[0]
        else:
            before_v1 = ""
        self.base_url: str = urlunparse(parsed_url[:2] + ("",) + ("",) * 3)
        self.v1_url: str = urlunparse(parsed_url[:2]+ (before_v1,) + ("",) * 3)
        self.v1_models: str = urlunparse(parsed_url[:2] + (before_v1 + "models",) + ("",) * 3)
        self.chat_url: str = urlunparse(parsed_url[:2] + (before_v1 + "chat/completions",) + ("",) * 3)
        self.image_url: str = urlunparse(parsed_url[:2] + (before_v1 + "images/generations",) + ("",) * 3)
        self.audio_transcriptions: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/transcriptions",) + ("",) * 3)
        self.moderations: str = urlunparse(parsed_url[:2] + (before_v1 + "moderations",) + ("",) * 3)
        self.embeddings: str = urlunparse(parsed_url[:2] + (before_v1 + "embeddings",) + ("",) * 3)
        self.audio_speech: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/speech",) + ("",) * 3)

def get_engine(provider, endpoint=None, original_model=""):
    parsed_url = urlparse(provider['base_url'])
    # print("parsed_url", parsed_url)
    engine = None
    stream = None
    if parsed_url.path.endswith("/v1beta") or parsed_url.path.endswith("/v1"):
        engine = "gemini"
    elif parsed_url.netloc == 'aiplatform.googleapis.com':
        engine = "vertex"
    elif parsed_url.netloc.rstrip('/').endswith('openai.azure.com') or parsed_url.netloc.rstrip('/').endswith('services.ai.azure.com'):
        engine = "azure"
    elif parsed_url.netloc == 'api.cloudflare.com':
        engine = "cloudflare"
    elif parsed_url.netloc == 'api.anthropic.com' or parsed_url.path.endswith("v1/messages"):
        engine = "claude"
    elif parsed_url.netloc == 'api.cohere.com':
        engine = "cohere"
        stream = True
    else:
        engine = "gpt"

    original_model = original_model.lower()
    if original_model \
    and "claude" not in original_model \
    and "gpt" not in original_model \
    and "deepseek" not in original_model \
    and "o1" not in original_model \
    and "o3" not in original_model \
    and "gemini" not in original_model \
    and "learnlm" not in original_model \
    and "grok" not in original_model \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in original_model and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in original_model and engine == "vertex":
        engine = "vertex-gemini"

    if provider.get("engine"):
        engine = provider["engine"]

    if endpoint == "/v1/images/generations" or "stable-diffusion" in original_model:
        engine = "dalle"
        stream = False

    if endpoint == "/v1/audio/transcriptions":
        engine = "whisper"
        stream = False

    if endpoint == "/v1/moderations":
        engine = "moderation"
        stream = False

    if endpoint == "/v1/embeddings":
        engine = "embedding"

    if endpoint == "/v1/audio/speech":
        engine = "tts"
        stream = False

    return engine, stream

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    if not data:
        return default
    return data

def parse_rate_limit(limit_string):
    # 定义时间单位到秒的映射
    time_units = {
        's': 1, 'sec': 1, 'second': 1,
        'm': 60, 'min': 60, 'minute': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
        'mo': 2592000, 'month': 2592000,
        'y': 31536000, 'year': 31536000
    }

    # 处理多个限制条件
    limits = []
    for limit in limit_string.split(','):
        limit = limit.strip()
        # 使用正则表达式匹配数字和单位
        match = re.match(r'^(\d+)/(\w+)$', limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {limit}")

        count, unit = match.groups()
        count = int(count)

        # 转换单位到秒
        if unit not in time_units:
            raise ValueError(f"Unknown time unit: {unit}")

        seconds = time_units[unit]
        limits.append((count, seconds))

    return limits

class ThreadSafeCircularList:
    def __init__(self, items = [], rate_limit={"default": "999999/min"}, schedule_algorithm="round_robin"):
        if schedule_algorithm == "random":
            import random
            self.items = random.sample(items, len(items))
            self.schedule_algorithm = "random"
        elif schedule_algorithm == "round_robin":
            self.items = items
            self.schedule_algorithm = "round_robin"
        elif schedule_algorithm == "fixed_priority":
            self.items = items
            self.schedule_algorithm = "fixed_priority"
        else:
            self.items = items
            logger.warning(f"Unknown schedule algorithm: {schedule_algorithm}, use (round_robin, random, fixed_priority) instead")
            self.schedule_algorithm = "round_robin"
        self.index = 0
        self.lock = asyncio.Lock()
        # 修改为二级字典，第一级是item，第二级是model
        self.requests = defaultdict(lambda: defaultdict(list))
        self.cooling_until = defaultdict(float)
        self.rate_limits = {}
        if isinstance(rate_limit, dict):
            for rate_limit_model, rate_limit_value in rate_limit.items():
                self.rate_limits[rate_limit_model] = parse_rate_limit(rate_limit_value)
        elif isinstance(rate_limit, str):
            self.rate_limits["default"] = parse_rate_limit(rate_limit)
        else:
            logger.error(f"Error ThreadSafeCircularList: Unknown rate_limit type: {type(rate_limit)}, rate_limit: {rate_limit}")

    async def set_cooling(self, item: str, cooling_time: int = 60):
        """设置某个 item 进入冷却状态

        Args:
            item: 需要冷却的 item
            cooling_time: 冷却时间(秒)，默认60秒
        """
        if item == None:
            return
        now = time()
        async with self.lock:
            self.cooling_until[item] = now + cooling_time
            # 清空该 item 的请求记录
            # self.requests[item] = []
            logger.warning(f"API key {item} 已进入冷却状态，冷却时间 {cooling_time} 秒")

    async def is_rate_limited(self, item, model: str = None) -> bool:
        now = time()
        # 检查是否在冷却中
        if now < self.cooling_until[item]:
            return True

        # 获取适用的速率限制

        if model:
            model_key = model
        else:
            model_key = "default"

        rate_limit = None
        # 先尝试精确匹配
        if model and model in self.rate_limits:
            rate_limit = self.rate_limits[model]
        else:
            # 如果没有精确匹配，尝试模糊匹配
            for limit_model in self.rate_limits:
                if limit_model != "default" and model and limit_model in model:
                    rate_limit = self.rate_limits[limit_model]
                    break

        # 如果都没匹配到，使用默认值
        if rate_limit is None:
            rate_limit = self.rate_limits.get("default", [(999999, 60)])  # 默认限制

        # 检查所有速率限制条件
        for limit_count, limit_period in rate_limit:
            # 使用特定模型的请求记录进行计算
            recent_requests = sum(1 for req in self.requests[item][model_key] if req > now - limit_period)
            if recent_requests >= limit_count:
                logger.warning(f"API key {item} 对模型 {model_key} 已达到速率限制 ({limit_count}/{limit_period}秒)")
                return True

        # 清理太旧的请求记录
        max_period = max(period for _, period in rate_limit)
        self.requests[item][model_key] = [req for req in self.requests[item][model_key] if req > now - max_period]

        # 记录新的请求
        self.requests[item][model_key].append(now)
        return False

    async def next(self, model: str = None):
        async with self.lock:
            if self.schedule_algorithm == "fixed_priority":
                self.index = 0
            start_index = self.index
            while True:
                item = self.items[self.index]
                self.index = (self.index + 1) % len(self.items)

                if not await self.is_rate_limited(item, model):
                    return item

                # 如果已经检查了所有的 API key 都被限制
                if self.index == start_index:
                    logger.warning(f"All API keys are rate limited!")
                    raise HTTPException(status_code=429, detail="Too many requests")

    async def after_next_current(self):
        # 返回当前取出的 API，因为已经调用了 next，所以当前API应该是上一个
        if len(self.items) == 0:
            return None
        async with self.lock:
            item = self.items[(self.index - 1) % len(self.items)]
            return item

    def get_items_count(self) -> int:
        """返回列表中的项目数量

        Returns:
            int: items列表的长度
        """
        return len(self.items)

def circular_list_encoder(obj):
    if isinstance(obj, ThreadSafeCircularList):
        return obj.to_dict()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

provider_api_circular_list = defaultdict(ThreadSafeCircularList)

# 【GCP-Vertex AI 目前有這些區域可用】 https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude?hl=zh_cn
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations?hl=zh-cn#available-regions

# c3.5s
# us-east5
# europe-west1

# c3s
# us-east5
# us-central1
# asia-southeast1

# c3o
# us-east5

# c3h
# us-east5
# us-central1
# europe-west1
# europe-west4


c35s = ThreadSafeCircularList(["us-east5", "europe-west1"])
c3s = ThreadSafeCircularList(["us-east5", "us-central1", "asia-southeast1"])
c3o = ThreadSafeCircularList(["us-east5"])
c3h = ThreadSafeCircularList(["us-east5", "us-central1", "europe-west1", "europe-west4"])
gemini1 = ThreadSafeCircularList(["us-central1", "us-east4", "us-west1", "us-west4", "europe-west1", "europe-west2"])
gemini2 = ThreadSafeCircularList(["us-central1"])