from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

chat_template = ChatPromptTemplate(
    [
        ('system', "You are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="messages"),
        ('human', '{query}')
    ]
)

chat_history=[]

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'chatHistory.txt')) as f:
    chat_history.extend(f.readlines())

print(chat_history)

result = chat_template.invoke({
    "messages": chat_history,
    "query": "What is the return policy?"
})

print(result)



