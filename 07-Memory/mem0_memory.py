import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from mem0 import Memory

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

config = {
    "version":"v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            # Explicitly set dimension to match what the model actually produces
            "embedding_dims": 768
        }
    },
    "llm":{
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        }
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "host":"localhost",
            "port":"6333",
            # Make sure Qdrant collection is configured for 768 dimensions
            "embedding_model_dims": 768
        }
    }
}

mem_client = Memory.from_config(config)



def chat():
    while True:
        user_query = input(" 👨 > ")
        relevant_memories = mem_client.search(query=user_query,user_id="pravin007")
        
        if user_query.lower() in ['quit', 'exit', 'bye','q']:
            print("👋 Goodbye!")
            break

        memories = [f"ID: {mem.get("id")} Memory: {mem.get("memory")}" for mem in relevant_memories.get("results")]

        SYSTEM_PROMPT = f"""
             You are an memeory aware assistant which responds to user with context.
            You are given with past memories and facts about the user.
            
            Memory of the user:
            {json.dumps(memories)}

            """

        try:
            model = genai.GenerativeModel(model_name="gemini-2.0-flash",system_instruction=SYSTEM_PROMPT)
            chat = model.start_chat()    
            response = chat.send_message(user_query)

            print(" 🤖 : ", response.text)

            # Add conversation to memory
            mem_client.add([
                {"role":"user","content":user_query},
                {"role":"assistant","content":response.text}
            ], user_id="pravin007")
            
        except Exception as e:
            print(f"❌ Error: {e}")
           



if __name__ == "__main__":
    chat()