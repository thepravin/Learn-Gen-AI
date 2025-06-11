import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API Key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# Configure the genai client
genai.configure(api_key=api_key)


#******** Chain Of Thought: The model is encouraged to break down reasoning step by step before arriving at an answer. ************

SYSTEM_PROMPT = """
   You are an helpfull AI assistant who is specialized in resolving user query.
    For the given user input, analyse the input and break down the problem step by step.

    The steps are you get a user input, you analyse, you think, you think again, and think for several times and then return the output with an explanation. 

    Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

    Rules:
    1. Follow the strict JSON output as per schema.
    2. Always perform one step at a time and wait for the next input.
    3. Carefully analyse the user query,

    Output Format:
    {{ "step": "string", "content": "string" }}

    Example:
    Input: What is 2 + 2
    Output: {{ "step": "analyse", "content": "Alight! The user is interest in maths query and he is asking a basic arthematic operation" }}
    Output: {{ "step": "think", "content": "To perform this addition, I must go from left to right and add all the operands." }}
    Output: {{ "step": "output", "content": "4" }}
    Output: {{ "step": "validate", "content": "Seems like 4 is correct ans for 2 + 2" }}
    Output: {{ "step": "result", "content": "2 + 2 = 4 and this is calculated by adding all numbers" }}
"""

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  
    system_instruction=SYSTEM_PROMPT,
    generation_config={
        "response_mime_type": "application/json"  # give response in json format.
    }
)


""" 
The issue is that chat_history is initialized as an empty list at the beginning and never gets updated. In Google Generative AI, when you use chat.send_message(), the chat object internally maintains its own history, but it doesn't automatically update your chat_history variable.
"""
chat_history = []

# Function to update chat history manually
def update_chat_history(role, content):
    chat_history.append({"role": role, "parts": [content]})


query = input("> ")

# Start chat session with the user query
chat = model.start_chat(history=chat_history)

try:
    # Add initial user query to history
    update_chat_history("user", query)
    
    while True:
        # Send message to Gemini
        response = chat.send_message(query)
        
        # Add response to history
        update_chat_history("model", response.text)
        
        # Parse the JSON response
        try:
            parsed_response = json.loads(response.text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw response: {response.text}")
            break
        
        # Handle different steps
        if parsed_response.get("step") == "think":
            print("   🧠 THINK :", parsed_response.get("content"))           
            query = "Continue to the next step."
            update_chat_history("user", query)
            continue
            
        elif parsed_response.get("step") == "analyse":
            print("   🔍 ANALYSE :", parsed_response.get("content"))
            query = "Continue to the next step."
            update_chat_history("user", query)
            continue
            
        elif parsed_response.get("step") == "output":
            print("    📤 OUTPUT :", parsed_response.get("content"))
            query = "Continue to the next step."
            update_chat_history("user", query)
            continue
            
        elif parsed_response.get("step") == "validate":
            print("    ✅ VALIDATE :", parsed_response.get("content"))
            query = "Continue to the next step."
            update_chat_history("user", query)
            continue
            
        elif parsed_response.get("step") == "result":
            print("\n\n 🤖 : ", parsed_response.get("content"),"\n\n")
            # print("Internal chat history:", chat.history)
            # print("\n\n📋 Chat History:")
            # for i, msg in enumerate(chat_history):
            #     print(f"{i+1}. {msg['role']}: {msg['parts'][0][:100]}...")  # Truncate long messages
            break
            
        else:
            # Handle unexpected steps
            print(f"          🔄 {parsed_response.get('step', 'unknown').upper()}:", parsed_response.get("content"))
            query = "Continue to the next step."
            update_chat_history("user", query)
            continue

except Exception as e:
    print(f"❌ Error: {e}")





