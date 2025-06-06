from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()

# Set up model
llm = HuggingFaceEndpoint(repo_id="google/gemma-2b-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

# Define response schema
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about topic")
]

# Create parser from schema
parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt template with format instructions
template1 = PromptTemplate(
    template="Give 3 facts about {topic}. {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Chain: Prompt → Model → Parse
chain = template1 | model | parser

# Run chain
result = chain.invoke({"topic": "blackhole"})

# Output
print(result)
