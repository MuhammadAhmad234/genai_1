from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)


def word_counter(text):
    return len(text.split(" "))


load_dotenv()


first_prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(first_prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_counter),
    }
)

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({"topic": "Donald Trump"})

print(result)
