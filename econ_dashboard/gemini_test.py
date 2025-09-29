from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = GEMINI_API_KEY)

response = client.models.generate_content(
    model="models/gemini-flash-latest",
    contents="how many countries are in the americas?",
    config=types.GenerateContentConfig(
        system_instruction="you are a Wall Street investment analyst and trader"
    )
)


print(response.text)