from ..utils import BaseAPI

import os

# GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)

# base_api = BaseAPI("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=" + GOOGLE_AI_API_KEY)

# print(base_api.chat_url)

base_api = BaseAPI("http://127.0.0.1:8000/v1")

print(base_api.chat_url)

"""
python -m core.test.test_base_api
"""