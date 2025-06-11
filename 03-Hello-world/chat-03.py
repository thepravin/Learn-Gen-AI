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
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT,
    generation_config={
        "response_mime_type": "application/json"  # give response in json format.
    }
)

# Start a chat session
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["Hey, My name is Pravin"]
    },
    {
        "role": "model", 
        "parts": ["Hi Pravin, it's nice to meet you! How can I help you today?"]
    },

    # add bot respones


    {
        "role":"model",
        "parts":[json.dumps({ "step": "analyse","content": "The user is asking to evaluate a mathematical expression that involves division, multiplication inside the parenthesis and then exponentiation. Order of operations (PEMDAS/BODMAS) must be followed to get the correct result."})]
    }
    ,
    {
        "role":"model",
        "parts":[json.dumps({"step": "think", "content": "I need to calculate the value of the expression (5/2*3)^4. First, I will perform the operations inside the parentheses from left to right. Then, I will raise the result to the power of 4."})]
    },
    {
        "role":"model",
        "parts": [ json.dumps({"step": "output", "content": "1.  Calculate 5 / 2 = 2.5\n2.  Calculate 2.5 * 3 = 7.5\n3.  Calculate 7.5 ^ 4 = 3164.0625\n\nTherefore, (5/2*3)^4 = 3164.0625"})]
    },
    {
        "role":"model",
        "parts":[json.dumps({"step": "validate", "content": "To validate the result, I can use a calculator to evaluate the expression (5/2*3)^4 and see if it matches the calculated value.\n(5/2*3)^4 = (2.5*3)^4 = (7.5)^4 = 3164.0625. The calculated value matches the result of the calculator"} )]
    },
    {
        "role":"model",
        "parts":[json.dumps({"step": "result", "content": "The value of the expression (5/2*3)^4 is 3164.0625"} )]
    }

])

# Send the new message
response = chat.send_message("What is 5/2*3 to the power 4")

# Print the response
print("\n\n🤖 bot: ", response.text,"\n\n")


