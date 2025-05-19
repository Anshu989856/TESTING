from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Initialize the model
llm = HuggingFaceEndpoint(repo_id="microsoft/phi-4")
model = ChatHuggingFace(llm=llm)

# Define chat prompt template using tuple format
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Initialize chat history
chat_history = []

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Format prompt with current chat history and user input
    messages = chat_template.format_messages(
        chat_history=chat_history,
        query=user_input  # 'query' matches the placeholder in the prompt
    )

    # Get response from model
    result = model.invoke(messages)

    # Print and update chat history
    print("AI:", result.content)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))
