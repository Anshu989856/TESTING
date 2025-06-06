from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

# Define the prompt
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

# Output parser
parser = StrOutputParser()

# Load the document
loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()

# Debug prints
print(type(docs))             # Should be list
print(len(docs))              # Number of documents

# Create the chain
chain = prompt | llm | parser

# Run the chain
result = chain.invoke({'poem': docs[0].page_content})
print(result)
