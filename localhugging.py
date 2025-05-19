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
