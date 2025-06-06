from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

# Load local Hugging Face model (e.g., Zephyr 7B)
chat_model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100}
)

# Output parser (string)
parser = StrOutputParser()

# Pydantic parser for classification
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
classifier_chain = prompt1 | chat_model | parser2

# Prompts for generating responses
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Conditional branching
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | chat_model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | chat_model | parser),
    RunnableLambda(lambda x: "Could not detect sentiment.")
)

# Final processing chain
chain = classifier_chain | branch_chain

# Example run
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print("Response:\n", result)

# Optional: Print graph
chain.get_graph().print_ascii()
