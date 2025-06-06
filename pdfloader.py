from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# Load HuggingFace model pipeline using GPU (optional: device=0) or CPU (device=-1)
hf_pipeline = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-128k-instruct",
    device=0,  # Set to -1 if you don't have a GPU
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)

# Wrap with LangChain-compatible pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt template
prompt = PromptTemplate(
    template="Write a summary for the following poem:\n\n{poem}",
    input_variables=["poem"]
)

# Output parser
parser = StrOutputParser()

# Load only the first page of the PDF
loader = PyPDFLoader("dl-curriculum.pdf")
first_page = loader.load()[0]  # Load and pick the first page

# Compose the LangChain pipeline
chain = prompt | llm | parser

# Run on the first page only
result = chain.invoke({"poem": first_page.page_content})

# Print summary
print("\n=== Summary of First Page ===\n")
print(result)
