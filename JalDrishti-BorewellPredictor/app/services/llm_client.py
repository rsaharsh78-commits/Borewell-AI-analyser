import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Centralized Featherless AI Client Initialization
featherless_client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY"),
)

def get_llm_client() -> OpenAI:
    """Returns the single initialized Featherless client instance."""
    return featherless_client