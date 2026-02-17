from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Tell me 5 interesting things about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Lionel Messi"})

print(result)
