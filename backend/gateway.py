import os
import logging
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("PORTKEY_API_KEY")
CONFIG_SLUG = os.environ.get("PORTKEY_CONFIG")
BASE_URL = os.environ.get("PORTKEY_BASE_URL")

def get_gateway_llm() -> ChatOpenAI:

    headers = createHeaders(api_key=API_KEY, config=CONFIG_SLUG)

    return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=BASE_URL,default_headers=headers)

