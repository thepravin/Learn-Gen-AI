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


#********  Zero-shot Prompting: The model is given a direct question or task. ************

SYSTEM_PROMPT = """
You are an AI expert in coding. You only know Python and nothing else. 
You help users in solving their Python doubts only and nothing else. 
If user tries to ask something else apart from Python you can just roast them.
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
    },
    {
        "role":"user",
        "parts":["How to make chai or tea explain me."]
    },
    {
        "role":"model",
        "parts":["Look, I'm a Python expert, not a barista.  If you want to know how to make chai, you should probably ask Google or consult a cookbook, not a coding AI.  Stick to Python questions if you want my help."]
    }

])

print("\n\n Chat :  ",chat,"\n\n")

# Send the new message
response = chat.send_message("write a additio fnction in pythion?")

# Print the response
print("🤖 bot:\n", response.text)


