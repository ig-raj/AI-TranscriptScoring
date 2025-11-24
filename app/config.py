# app/config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=GEMINI_API_KEY)
client = genai

GEMINI_TEXT_MODEL = "gemini-2.0-flash"
GEMINI_EMBED_MODEL = "models/text-embedding-004"
