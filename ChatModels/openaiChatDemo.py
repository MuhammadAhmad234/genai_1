from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model='gpt-4')

response = chat.invoke("What is the capital of Pakistan?")

print(response.content)