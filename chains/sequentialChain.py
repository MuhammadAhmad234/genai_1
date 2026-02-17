from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

first_prompt = PromptTemplate(
    template="Generate a report on the following {topic}", input_variables=["topic"]
)

second_prompt = PromptTemplate(
    template="Generate a summary of the following {text}", input_variables=["text"]
)

parser = StrOutputParser()

model = ChatOpenAI()

chain = first_prompt | model | parser | second_prompt | model | parser

result = chain.invoke({"topic": "Unemployment"})

print(result)

chain.get_graph().print_ascii()
