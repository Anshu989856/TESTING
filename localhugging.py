from langchain_huggingface import HuggingFacePipeline

# Load a small, public model
llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",                      # Smaller GPT2 variant
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100}
)

# Direct usage
response = llm.invoke("Hannibal")
print(response)

from langchain_huggingface import ChatHuggingFace

chat_model = ChatHuggingFace.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100}
)
response = chat_model.invoke([HumanMessage(content="Who was Hannibal and will graham?")])
print(response.content)
