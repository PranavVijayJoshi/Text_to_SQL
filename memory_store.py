from langgraph.store.memory import InMemoryStore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

store = InMemoryStore(
    index={
        "dims":768,
        "embed":embeddings
    }
)



