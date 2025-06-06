from langchain_huggingface import ChatHuggingFace

# Load a chat-based model
chat_model = ChatHuggingFace.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",  # Can be "text-generation" or "text2text-generation"
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 150,
        "do_sample": True
    }
)

# Chat format message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who was Hannibal and why is he famous?"}
]

# Invoke the model with the chat messages
response = chat_model.invoke(messages)
print(response)
