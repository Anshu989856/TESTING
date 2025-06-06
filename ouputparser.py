from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load environment variables
load_dotenv()

# Load LLM
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

# Define templates
template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Write a 5 line summary on {text}", input_variables=["text"])

parser=StrOutputParser()
chain=template1|model|parser|template2|model|parser
result=chain.invoke({"topic":"blackhole"})
print(result)