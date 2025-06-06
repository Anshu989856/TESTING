from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load models
llm1 = HuggingFaceEndpoint(repo_id="microsoft/Phi-3.5-mini-instruct", task="text-generation")
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(repo_id="microsoft/Phi-3.5-mini-instruct", task="text-generation")
model2 = ChatHuggingFace(llm=llm2)

# Prompt templates
prompt1 = PromptTemplate(template="Generate short and simple notes about: {text}", input_variables=["text"])
prompt2 = PromptTemplate(template="Generate 5 short question-answer pairs based on: {text}", input_variables=["text"])
prompt3 = PromptTemplate(template="Merge these notes:\n{notes}\n\nAnd these quiz questions:\n{quiz}", input_variables=["notes", "quiz"])

# Parser
parser = StrOutputParser()

# Parallel execution chain
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

# Merge chain
merge_chain = prompt3 | model1 | parser

# Final chain: run parallel first, then merge
full_chain = parallel_chain | merge_chain

# Example input text
text = "Photosynthesis is the process by which green plants convert sunlight into energy using carbon dioxide and water."

# Run the chain
result = full_chain.invoke({"text": text})

print(result)
full_chain.get_graph().printascii()