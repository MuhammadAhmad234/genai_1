from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="Fact 1", description="Fact number 1"),
    ResponseSchema(name="Fact 2", description="Fact number 2"),
    ResponseSchema(name="Fact 3", description="Fact number 3"),
    ResponseSchema(name="Fact 4", description="Fact number 4"),
    ResponseSchema(name="Fact 5", description="Fact number 5"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me five facts about {topic}. /n {format_instructions}",
    input_variables = ["topic"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'AI'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
