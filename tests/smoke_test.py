import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config
from utils import llm
from utils.errors import ProviderError
from modules import seo_research


def check(name, fn):
    try:
        print(f"PASS  {name}: {fn()}")
    except ProviderError as e:
        print(f"FAIL  {name}: [{type(e).__name__}] {e}")


config.validate_keys()

print("--- Real keys ---")
check("Gemini text", lambda: llm.ask("Reply with the single word: ok", max_tokens=50).strip())
check("Gemini JSON", lambda: llm.ask_json('Return {"status": "ok"}'))
check("Tavily search", lambda: f"{len(seo_research.tavily_search('email marketing tips', 3))} results")

print("\n--- Deliberately bad keys (expect InvalidKeyError messages) ---")
real_g, real_t = config.GEMINI_API_KEY, config.TAVILY_API_KEY

config.GEMINI_API_KEY = ""
llm._client = None
check("Gemini bad key", lambda: llm.ask("hi", max_tokens=20))

config.TAVILY_API_KEY = ""
seo_research._tavily = None
check("Tavily bad key", lambda: seo_research.tavily_search("test", 1))

config.GEMINI_API_KEY, config.TAVILY_API_KEY = real_g, real_t
