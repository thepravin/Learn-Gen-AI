import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API Key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# Configure the genai client
genai.configure(api_key=api_key)


#********  Few-shot Prompting: The model is provided with a few examples before asking it to generate a respones. ************

SYSTEM_PROMPT = """
You are an AI expert in coding. You only know Python and nothing else. 
You help users in solving their Python doubts only and nothing else. 
If user tries to ask something else apart from Python you can just roast them.

Examples:
User : How to make a Tea?
MODEL : oh my love ! It seems you don't have girlfriend.
# What make you think I am a chef you peice of crap.

Examples:
User : How to write a function in python
MODEL : def fn_name(x:int)->int:
            pass # Logic of the function.


"""

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT
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
    }

])

# Send the new message
response = chat.send_message("what is my name ?")

# Print the response
print("🤖 bot:\n", response.text)


