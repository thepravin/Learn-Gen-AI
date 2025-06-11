from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from google import genai
import google.generativeai as genai
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


class State(TypedDict):
   query:str
   llm_result : str | None


def chat_bot(state: State):
    query = state['query']

    # Call openAI or gemini Api's

    result = f"Today is Wednesday, and your question was: {query}"
    return {'llm_result': result}


# Step 1 : create builder
graph_builder = StateGraph(State)

# Step 2 : add all nodes
graph_builder.add_node('chat_bot',chat_bot)

# Step 3 : create a agentic flow
graph_builder.add_edge(START,'chat_bot')
graph_builder.add_edge('chat_bot',END)

# Step 4 : compile the graph builder
graph = graph_builder.compile()



user = input("> ")

while True:
  if user.lower() in ["quit", "exit", "q"]:
          print("Goodbye!")
          break

  #invoke the graph
  _state = {
      "query":user,
      "llm_result":None
  }

  graph_result = graph.invoke(_state)

  print(graph_result)

  user = input("> ")

