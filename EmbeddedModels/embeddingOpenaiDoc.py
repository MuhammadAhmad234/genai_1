from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model= "text-embedding-3-small", dimensions=32)

document =[
    "Islamabad is the capital of pakistan",
    "Lahore is Capital of Punjab",
    "Peshawar is Capital of KPK",
    "Karachi is Capital of Sindh"
]

vector = model.embed_documents(document)
print(str(vector))