from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

st.header('Research Tool')

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Select....", "Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select ExplanationStyle",
    ["Select....", "Beginner Friendly", "Technical", "Code Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Length",
    ["Select....", "Short(1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long(Detailed explanation)"]
)

template = load_prompt("template.json")





if st.button("Summarize"):
    chain = template | model
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        }
    )
    st.write(result.content)


