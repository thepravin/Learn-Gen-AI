import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
text = "Hello, I am Pravin"
token = enc.encode(text)

print("Tokens : ",token)

decoded = enc.decode(token)
print("Decoded : ",decoded)