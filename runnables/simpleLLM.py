
# Sample LLM Application (Old Way when there were no chat models)
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

prompt = PromptTemplate(
    input_variable=["topic"], template="Suggest a catchy blog title about {topic}"
)

topic = input("Enter a topic: ")

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print(blog_title)
