from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

client = OpenAI()

text = "dog and cat"

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

print(response)