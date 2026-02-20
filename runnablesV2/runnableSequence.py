from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence


load_dotenv()

model = ChatOpenAI()

first_prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

second_prompt = PromptTemplate(
    template="Explain this  {joke}", input_variables=["joke"]
)

parser = StrOutputParser()

chain = RunnableSequence(first_prompt, model, parser, second_prompt, model, parser)

result = chain.invoke({"topic": "Lionel Messi"})

print(result)
