from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
import google.generativeai as genai
from typing import Annotated
from dotenv import load_dotenv
import os
import json
from langgraph.types import interrupt,Command


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------------------------------------------------------------

class State(TypedDict):
    messages : Annotated[list, add_messages]

# ---------------------------------- Tools ---------------------------------------------------

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]



tools_fun = [human_assistance]

# ----------------------------------- llm models -----------------------------------------------

llm = init_chat_model("google_genai:gemini-2.0-flash",model_kwargs={"response_format": "text"})
llm_with_tools = llm.bind_tools(tools_fun)


# ------------------------------- Nodes ------------------------------------------------------

def chatbot (state:State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages":[message]}


tool_node = ToolNode(tools = tools_fun)

# ---------------------------------- Graph --------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)


graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)


graph = graph_builder.compile()


# ---------------------------------- ceckpointer ---------------------------------
def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer

# -------------------------- main -------------------------------------------------

def user_call():
    #   mongodb://<username>:<pass>@<host>:<port>
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = { "configurable":{"thread_id":"21"}}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

        while True:
            query = input("\n 👨 >  ")

            state = State(
                messages=[{"role": "user", "content": query}]
            )

            for event in graph_with_mongo.stream(state,config,stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()
            


def admin_call():
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = { "configurable":{"thread_id":"21"}}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

        state = graph_with_mongo.get_state(config=config)
        
        last_message = state.values["messages"][-1]
       

        tool_calls = last_message.additional_kwargs.get("function_call",[])
        # print(tool_calls)

        user_query = None

      
        # Normalize to list
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]

        
        for call in tool_calls:
            if call.get('name') == "human_assistance":
                args = call.get("arguments", "{}")
                try:
                    args_dict = json.loads(args)
                    user_query = args_dict.get("query")
                    print(f"User query: {user_query}")
                except json.JSONDecodeError:
                    print("Failed to decode function arguments.")
        
        # print("User Has a Query", user_query)
        solution = input("> ")

        resume_command = Command(resume={"data": solution})

        for event in graph_with_mongo.stream(resume_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()

        

# -------------------------------------------------------------------------------------------------


user_call()
# admin_call()