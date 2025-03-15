"""
uni-api-core 模块
提供统一的API请求和处理功能
"""

# 从models.py导入主要类
from .models import (
    RequestModel,
    ImageGenerationRequest,
    AudioTranscriptionRequest,
    ModerationRequest,
    TextToSpeechRequest,
    UnifiedRequest,
    EmbeddingRequest,
    Message,
    Tool,
    Function
)

# 从request.py导入主要函数
from .request import (
    get_payload,
    encode_image
)

# 从utils.py导入主要工具函数和类
from .utils import (
    safe_get,
    get_model_dict,
    ThreadSafeCircularList,
    get_engine,
    BaseAPI
)

__version__ = "1.0.0"