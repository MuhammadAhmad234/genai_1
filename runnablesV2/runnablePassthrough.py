from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough


load_dotenv()


first_prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

second_prompt = PromptTemplate(
    template="Explain this joke {joke}", input_variables=["joke"]
)

joke_generator_chain = RunnableSequence(first_prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(second_prompt, model, parser),
    }
)

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({"topic": "Donald Trump"})

print(result)
