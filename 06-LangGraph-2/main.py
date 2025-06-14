import os
from typing import Annotated
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]



def chat_node(state: State):
    model = genai.GenerativeModel(
        model_name= "gemini-2.0-flash",        
        generation_config={
            "response_mime_type":"application/json",            
        }
        )

    chat = model.start_chat()    

    # Extract user message content (string)
    user_msg = state["messages"][-1].content  # Assuming last message is from user

    response = chat.send_message(user_msg)
    # return {"messages": [response.text]} 
    return {"messages": [AIMessage(content=response.text)]}
    

graph_builder = StateGraph(State)
graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

graph = graph_builder.compile()

def main():
    query = input(" 👨 > ")
    result = graph.invoke({"messages": [{"role": "user", "content": query}]})
    print(result)

main()
