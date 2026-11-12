import os
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("PORTKEY_API_KEY")
CONFIG_SLUG = os.environ.get("PORTKEY_CONFIG")
BASE_URL = os.environ.get("PORTKEY_BASE_URL")


def get_gateway_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Return an LLM instance, preferring Portkey if available, falling back directly to OpenAI."""
    if API_KEY and BASE_URL:
        try:
            from portkey_ai import createHeaders
            headers = createHeaders(api_key=API_KEY, config=CONFIG_SLUG)
            return ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                base_url=BASE_URL,
                default_headers=headers,
                model=model,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Portkey init failed: %s. Falling back to direct OpenAI.", e)

    return ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=model,
        temperature=temperature,
    )
