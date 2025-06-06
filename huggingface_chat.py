from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="facebook/blenderbot-400M-distill",task="text-generation")
model=ChatHuggingFace(llm=llm)
print(model.invoke("Hannibal"))
