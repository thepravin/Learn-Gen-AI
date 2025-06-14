import os
import json
from typing import Annotated
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langgraph.checkpoint.mongodb import MongoDBSaver


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------------------------------------------

def chat_node(state: State):
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "response_mime_type": "application/json",
        }
    )

    # Convert LangChain messages to Gemini-compatible format
    gemini_messages = []
    for msg in state["messages"]:
        if msg.type == "human":
            gemini_messages.append({"role": "user", "parts": [msg.content]})
        elif msg.type == "ai":
            gemini_messages.append({"role": "model", "parts": [msg.content]})
        else:
            print(f"⚠️ Skipping unsupported message type: {msg.type}")

    # Now pass the formatted messages
    response = model.generate_content(gemini_messages)

    return {"messages": [AIMessage(content=response.text)]}


    

# -----------------------------------------------------------------


graph_builder = StateGraph(State)
graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

graph = graph_builder.compile()



def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer


# -----------------------------------------------------------------

def print_last_ai_message(state):
    messages = state.get("messages", [])
    # Filter AI messages
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    
    if ai_messages:
        last_ai = ai_messages[-1]
        content_dict = json.loads(last_ai.content)
        print(" 🤖 > ", content_dict.get("response"))
    else:
        print("No AI message found.")
      

def main():

    #   mongodb://<username>:<pass>@<host>:<port>
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = { "configurable":{"thread_id":"2"}}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:

        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

        query = input(" 👨 >  ")

        result = graph_with_mongo.invoke({"messages": [{"role": "user", "content": query}]},config)
        print_last_ai_message(result)
        # print(result)

    

main()
