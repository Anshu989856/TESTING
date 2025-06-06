from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Optional

# Initialize the Hugging Face LLM
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
model = ChatHuggingFace(llm=llm)

# Define structured output schema using Annotated with TypedDict
class Review(TypedDict):
    key_themes: Annotated[list[str], "Themes discussed"]
    summary: Annotated[str, "Brief summary"]
    sentiment: Annotated[str, "Positive, Negative, Neutral, Depends on Perspective"]
    pros: Annotated[Optional[str], "Pros"]

# Enable structured output
structured_model = model.with_structured_output(Review)

# Run the model
result = structured_model.invoke("Will Graham is a friend of Hannibal")
print(result)
