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


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------- Tools -----------------------------------------

@tool
def get_weather(city: str):
    """ This tool return the weather data about the given city """

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    
    return "Something went wrong."


tools_fun = [get_weather]



# -------------------------------- model -------------------------------------------------------

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools_fun)

def chat_node(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}
    
# ---------------------------------------------------------------------------------------------------

graph_builder = StateGraph(State)
graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
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
