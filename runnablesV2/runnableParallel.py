from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

first_prompt = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)

second_prompt = PromptTemplate(
    template="Generate a linkedin post about {topic}", input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(first_prompt, model, parser),
        "linkedin": RunnableSequence(second_prompt, model, parser),
    }
)

result = parallel_chain.invoke({"topic": "Generative AI"})

print(result)
