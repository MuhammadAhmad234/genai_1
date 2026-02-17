from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
first_parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


second_parser = PydanticOutputParser(pydantic_object=Feedback)

first_prompt = ChatPromptTemplate.from_template(
    "Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}",
    partial_variables={"format_instructions": second_parser.get_format_instructions()},
)

second_prompt = ChatPromptTemplate.from_template(
    "Write an appropriate response for this positive feedback \n {feedback}"
)

third_prompt = ChatPromptTemplate.from_template(
    "Write an appropriate response for this negative feedback \n {feedback}"
)


classifier_chain = first_prompt | model | second_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", second_prompt | model | first_parser),
    (lambda x: x.sentiment == "negative", third_prompt | model | first_parser),
    RunnableLambda(lambda x: "Could not find a sentiment"),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I love this product"})

print(result)
