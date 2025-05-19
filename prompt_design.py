import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Get Hugging Face API token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the Hugging Face LLM with explicit parameters
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=1024
)

# Wrap the LLM in ChatHuggingFace for chat-like interface
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title('🧠 Research Paper Summarizer')

paper_input = st.selectbox(
    "📄 Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "🎨 Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "📏 Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Prompt template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

# When user clicks the button
if st.button('📝 Summarize'):
    # Create prompt
    final_prompt = template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    # Call the model and display result
    result = model.invoke(final_prompt)
    st.subheader("🔍 Summary")
    st.write(result.content)
