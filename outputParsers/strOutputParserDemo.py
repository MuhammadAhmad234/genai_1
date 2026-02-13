from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)

first_template = PromptTemplate(
    template = 'Write a detailed prompt on {topic}',
    input_variables = ['topic']
)

second_template = PromptTemplate(
    template = 'Write a five line summary of the following text. /n {text}',
    input_variables = ['text']
)

first_prompt = first_template.invoke({'topic': 'AI'})
first_result = model.invoke(first_prompt)

print("Complete Text: ", first_result.content)

second_prompt = second_template.invoke({'text': first_result.content})

second_result = model.invoke(second_prompt)

print("Summarized Text: ", second_result.content)