from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
import json
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


class State(TypedDict):
  query : str
  llm_result : str | None
  accuracy_percentage : str | None
  is_coding_question : bool | None


# validate/ set a schema of AI respose
class classifyMessageResponse(BaseModel):
  is_coding_question : bool


class classifyCodeAccuracy(BaseModel):
  accuracy_percentage : str

# ------------- functions -----------------------

def classify_message(state: State):
    print("     ⚠️  Classify Message.....")

    query = state["query"]

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is
    related to coding question or not.
    Return the response in specified JSON boolean only.
    Example 1:
    {
      "is_coding_question": true
    }
    Example 2:
    {
      "is_coding_question": false
    }
    """

    model = genai.GenerativeModel(
        model_name= "gemini-2.0-flash",
        system_instruction= SYSTEM_PROMPT,
        generation_config={
            "response_mime_type":"application/json",
            "response_schema":classifyMessageResponse,
        }
        )

    chat = model.start_chat()
    response = chat.send_message(query)


    raw_text = response.text.strip()
    # print("Raw response:", raw_text)


    parsed = json.loads(raw_text)
    # print("Parsed response:", parsed)

    is_coding_question = parsed["is_coding_question"]
    state["is_coding_question"] = is_coding_question


    # return {"is_coding_question": is_coding_question}
    return state


def general_query(state: State):
    print("     ⚠️  general_query")
    query = state["query"]

    model = genai.GenerativeModel(
        model_name= "gemini-2.0-flash",
      )

    chat = model.start_chat()
    response = chat.send_message(query)
    llm_result = response.text.strip()

    state["llm_result"] = llm_result

    return state


def coding_query(state: State):
    print("     ⚠️  coding_query")
    query = state["query"]

    SYSTEM_PROMPT = """
        You are a Coding Expert Agent. Write a best and optimized code.
    """

    model = genai.GenerativeModel(
        model_name= "gemini-2.0-flash",
        system_instruction= SYSTEM_PROMPT,
        )

    chat = model.start_chat()
    response = chat.send_message(query)
    llm_result = response.text.strip()

    state["llm_result"] = llm_result

    return state


def route_query(state: State) -> Literal["general_query", "coding_query"]:
    print("     ⚠️  route_query")
    is_coding_question = state["is_coding_question"]
    if is_coding_question:
        return "coding_query"
    else:
        return "general_query"
    

# ------------------------- graph ---------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node('classify_message',classify_message)
graph_builder.add_node('general_query',general_query)
graph_builder.add_node('route_query',route_query)
graph_builder.add_node('coding_query',coding_query)


graph_builder.add_edge(START,'classify_message')
graph_builder.add_conditional_edges('classify_message', route_query)

graph_builder.add_edge("general_query", END)
graph_builder.add_edge("coding_query", END)


graph = graph_builder.compile()

# --------------------------- main ------------------------------------------

user = input(" 👨 > ")

while True :

    if user.lower() in ['quit','exit','q','e']:
       print("\n GoodBye 👋. . .\n")
       break

    _state: State = {
        "query": user,
        "accuracy_percentage": None,
        "is_coding_question": None,
        "llm_result": None
        }

    response = graph.invoke(_state)
    # print(json.dumps(response, indent=2))


    print(" 🤖 > ",response["llm_result"])

    user = input("\n 👨 > ")

