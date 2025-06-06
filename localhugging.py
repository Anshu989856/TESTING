from langchain_huggingface import HuggingFacePipeline

# Load a small, public model
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 150}
)

# Direct usage
response = llm.invoke("Hannibal")
print(response)

