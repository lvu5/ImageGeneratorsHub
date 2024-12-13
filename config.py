import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"
FALLBACK_ENGINE = os.getenv("FALLBACK_ENGINE", "")  # e.g., "runwayml"
