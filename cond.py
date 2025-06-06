from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# Load HuggingFace model (smaller & faster, free-tier)
endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-1.1-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=endpoint)

# Output parser
parser = StrOutputParser()

# Pydantic parser
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative:\n{feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Classifier chain
classifier_chain = prompt1 | model | parser2

# Prompts for appropriate response
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Branch based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not detect sentiment.")
)

# Final chain
chain = classifier_chain | branch_chain

# Example run
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)

# Show pipeline graph
chain.get_graph().print_ascii()
