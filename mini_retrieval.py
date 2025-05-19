from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (make sure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()

# Sample documents and query
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills and presence of mind.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query = 'tell me about bumrah'

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
doc_embeddings = embedding_model.embed_documents(documents)
query_embedding = embedding_model.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Output the result
print(query)
print(documents[index])
print("Similarity score is:", score)
