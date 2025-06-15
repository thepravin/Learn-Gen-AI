import os
import requests
import json
from typing import Annotated
from dotenv import load_dotenv
import google.generativeai as genai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------- Tools -----------------------------------------

@tool
def get_weather(city: str) -> dict:
    """
    This tool returns the temperature in Celsius for a given city.
    """
    url = f"https://wttr.in/{city}?format=%t"
    response = requests.get(url)

    if response.status_code == 200:
        temp_str = response.text.strip()
        try:
            # Extract number, e.g., "+32°C" -> 32
            temp = int(''.join(filter(lambda c: c.isdigit() or c == '-', temp_str)))
            return {"city": city, "temp_celsius": temp}
        except:
            return {"city": city, "temp_celsius": None, "error": "Could not parse temperature"}
    return {"city": city, "temp_celsius": None, "error": "Request failed"}


@tool
def add_two_numbers(a:int, b:int):
    """This tool adds two int numbers"""
    return a+b



tools_fun = [get_weather,add_two_numbers]



# -------------------------------- model -------------------------------------------------------

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools_fun)

# -------------------------- nodes / functions ---------------------------------------------------

def chat_node(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


tool_node = ToolNode(tools=tools_fun)
    
# ---------------------------------------------------------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_conditional_edges("chat_node",tools_condition)
graph_builder.add_edge("tools", "chat_node") # Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("chat_node", END)

graph = graph_builder.compile()

# -----------------------------------------------------------------------------------------------------

def main():
    query = input(" 👨 > ")

    state = State(
        messages= [{"role": "user", "content": query}]
    )

    for event in graph.stream(state,stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

# -----------------------------------------------------------------------------------------------------

main()
