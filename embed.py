from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Your Hugging Face API key must be in the environment as HUGGINGFACEHUB_API_TOKEN
embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed the query using the Hugging Face Inference API
embedding = embedding_model.embed_query("Hannibal season 2 is intriguing season 3 not so sure")
print(embedding)
