from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

# Load environment variables from .env file
load_dotenv()

# Corrected initialization of HuggingFaceEndpoint (note the equals sign and proper quotes)
llm = HuggingFaceEndpoint(repo_id="microsoft/phi-4")
# Initialize the ChatHuggingFace model
model = ChatHuggingFace(llm=llm)
messages=[SystemMessage(content="you are a helpful assistant"),
HumanMessage(content="Tell me about Langchain")]
# Chat loop
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
