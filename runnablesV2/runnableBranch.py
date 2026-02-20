from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
)


load_dotenv()


first_prompt = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

second_prompt = PromptTemplate(
    template="Summarize the following {text}", input_variables=["text"]
)

model = ChatOpenAI()

parser = StrOutputParser()

resport_generator_chain = RunnableSequence(first_prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(second_prompt, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(resport_generator_chain, branch_chain)

result = final_chain.invoke({"topic": "Generative AI"})

print(result)
