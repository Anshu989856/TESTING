from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from environment variables
# Ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables. "
                     "Please set it in your .env file.")

# Recommended model: TinyLlama-1.1B-Chat-v1.0
# Other options: google/gemma-2b-it, microsoft/Phi-3-mini-4k-instruct (check free tier suitability)
# The repo_id of the model you want to use.
# Make sure the model is publicly available on Hugging Face Hub.
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# repo_id = "google/gemma-2b-it" # Another good option
# repo_id = "microsoft/Phi-3-mini-4k-instruct" # Potentially good, but larger

try:
    print(f"Using model: {repo_id}")
    # Initialize the HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",  # Common task for chat models
        huggingfacehub_api_token=huggingface_api_token,
        max_new_tokens=150,  # Adjust as needed for response length
        # temperature=0.7,     # Adjust for creativity (optional)
        # top_p=0.95,          # Adjust for nucleus sampling (optional)
    )

    # Initialize the ChatHuggingFace wrapper
    model = ChatHuggingFace(llm=llm)

    # Test invocation
    print("Sending prompt to the model...")
    # For Chat models, invoke expects a string or a list of messages.
    # A simple string is treated as a single user message.
    response = model.invoke("Hannibal")

    # The response object from ChatHuggingFace is usually an AIMessage object.
    # You'll want to access its 'content' attribute.
    print("\nModel Response:")
    if hasattr(response, 'content'):
        print(response.content)
    else:
        # Fallback if the response structure is different (e.g., a raw string, though less likely with ChatHuggingFace)
        print(response)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Troubleshooting tips:")
    print("1. Verify your HUGGINGFACEHUB_API_TOKEN is correct and has 'read' permissions.")
    print(f"2. Check if the model '{repo_id}' is valid and publicly available on Hugging Face Hub.")
    print("3. Ensure the Hugging Face Inference API is not experiencing issues.")
    print("4. Your free tier credits might be exhausted.")