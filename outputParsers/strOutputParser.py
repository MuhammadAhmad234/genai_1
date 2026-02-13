from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

first_template = PromptTemplate(
    template="Write a detailed prompt on {topic}", input_variables=["topic"]
)

second_template = PromptTemplate(
    template="Write a five line summary of the following text. /n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = first_template | model | parser | second_template | model | parser

result = chain.invoke({"topic": "Blackhole"})

print(result)
