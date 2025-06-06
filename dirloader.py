from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables (OPENROUTER_API_KEY must be set here or in .env)
load_dotenv()

# Webpage to extract context
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)
docs = loader.load()

# Prompt template
prompt = PromptTemplate(
    template='Answer the following question:\n\n{question}\n\nBased on the text:\n\n{text}',
    input_variables=['question', 'text']
)

# Connect to OpenRouter model (using OpenAI-compatible interface)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-r1-0528:free",  
)

# Compose the chain
chain = prompt | llm | StrOutputParser()

# Ask a question
result = chain.invoke({
    'question': 'What is the product that we are talking about?',
    'text': docs[0].page_content
})

# Show result
print("\n=== Answer ===\n")
print(result)
