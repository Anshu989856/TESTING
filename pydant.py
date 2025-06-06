from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Set up model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2b-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

# Define the Pydantic model
class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(gt=18, description="Age of person (must be >18)")

# Create parser from schema
parser = PydanticOutputParser(pydantic_object=Person)

# Prompt template with format instructions
template1 = PromptTemplate(
    template="Give name and age of an {topic} person. {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Chain: Prompt → Model → Parse
chain = template1 | model | parser

# Run chain
result = chain.invoke({"topic": "Indian"})

# Output
print(result)
