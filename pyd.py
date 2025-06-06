from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
import json

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
model = ChatHuggingFace(llm=llm)

# Schema for validation
class Review(BaseModel):
    key_themes: List[str]
    summary: str
    sentiment: Literal["pos", "neg", "neutral"]
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    name: Optional[str] = None

# Prompt for JSON output
prompt = """
Given the following product review, extract the following details in JSON format with keys:
- key_themes (list of strings)
- summary (brief string)
- sentiment ("pos", "neg", or "neutral")
- pros (list of strings)
- cons (list of strings)
- name (reviewer's name)

Review:
"""
review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

full_prompt = prompt + review_text + "\n\nRespond in JSON only."

# Generate response
response = model.invoke(full_prompt)

# Try to parse and validate
try:
    parsed = json.loads(response.content if hasattr(response, 'content') else response)
    validated = Review(**parsed)
    print(validated.json(indent=2))
except (json.JSONDecodeError, ValidationError) as e:
    print("Error parsing or validating the response:\n", e)
    print("Raw model output:\n", response)
