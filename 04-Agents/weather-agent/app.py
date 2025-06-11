import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import requests
from datetime import datetime

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API Key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# Configure the genai client
genai.configure(api_key=api_key)


def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    
    return "Something went wrong."


# tool mapping
available_tools = {
    "get_weather": get_weather,
}

SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who is specialized in resolving user queries.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.

    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - "get_weather": Takes a city name as an input and returns the current weather for the city

    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interested in weather data of new york" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "content": "Based on the tool output, I can now provide the weather information" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

"""

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  
    system_instruction=SYSTEM_PROMPT,
    generation_config={
        "response_mime_type": "application/json"  # give response in json format.
    }
)

chat_history = []

""" query = input("> ")

# Start chat session with the user query
chat = model.start_chat(history=chat_history) """

# Start chat session with the user query
chat = model.start_chat(history=chat_history)

try:

    while True:
        query = input("> ")

        

        while True:
            # Send message to Gemini
            response = chat.send_message(query)        
            
            # Parse the JSON response
            try:
                parsed_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"Raw response: {response.text}")
                break
            
            # Handle different steps
            step = parsed_response.get("step")
            
            if step == "plan":
                print("   🧠 : ", parsed_response.get("content"))           
                query = "Continue to the next step."            
                continue
                
            elif step == "action":
                # print("\n\n Parsed_response : ",parsed_response,"\n\n")
                tool_name = parsed_response.get("function")
                tool_input = parsed_response.get("input")

                print(f"  🔨 : Calling Tool : {tool_name} with input: {tool_input}")

                # Fixed tool validation and execution
                if tool_name in available_tools:
                    try:
                        tool_output = available_tools[tool_name](tool_input)
                        print(f"  📊 : Tool output: {tool_output}")
                        # Send the tool output back to the model
                        query = f"Tool output: {tool_output}. Continue to the next step."
                    except Exception as e:
                        print(f"❌ Tool execution error: {e}")
                        query = f"Tool execution failed with error: {e}. Continue to the next step."
                else:
                    print(f"❌ Tool '{tool_name}' not found in available tools")
                    query = f"Tool '{tool_name}' not available. Continue to the next step."
                continue
                
            elif step == "observe":
                print("  👁️ : ", parsed_response.get("content"))
                query = "Continue to the next step."
                continue
                
            elif step == "output":
                print("🤖 : ", parsed_response.get("content"))
                break
                
            else:               
                print(f"❌ Unknown step: {step}")
                break

except Exception as e:
    print(f"❌ Error: {e}")