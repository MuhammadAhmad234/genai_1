from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = OpenAIEmbeddings(model= "text-embedding-3-large", dimensions=300)

document = [
    "Babar Azam is known for his elegant batting technique and consistent performances across all formats.",
    "Shaheen Afridi brings fiery pace and deadly swing, making him a nightmare for top-order batsmen.",
    "Wasim Akram revolutionized fast bowling with his lethal reverse swing and match-winning spells.",
    "Imran Khan led Pakistan to its historic 1992 World Cup victory with inspiring leadership.",
    "Shahid Afridi thrilled fans worldwide with his explosive hitting and aggressive style of play."
]


query = "Who led Pakistan to world cup victory?"

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print(query)
print(document[index])
print("Similarity score is", score)

