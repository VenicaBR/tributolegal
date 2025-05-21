from dotenv import load_dotenv
import os

load_dotenv()
print("Chave OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
