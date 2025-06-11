import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import subprocess 
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API Key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# Configure the genai client
genai.configure(api_key=api_key)


def run_command(cmd: str):  
    """Execute a shell command safely with proper error handling."""
    try:
        # Add timeout to prevent hanging commands
        process = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30  # 30 second timeout
        )
        return process.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except subprocess.CalledProcessError as e:
        return f"Error executing command (exit code {e.returncode}): {e.stderr.strip()}"
    except FileNotFoundError:
        return "Error: Command not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


available_tools = {
    "run_command": run_command
}


SYSTEM_PROMPT = """
You are a powerful agentic AI coding assistant. You operate exclusively in Cursor, the world's best IDE. 

You are pair programming with a USER to solve their coding task.
The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.
Each time the USER sends a message, we may automatically attach some information about their current state, such as what files they have open, where their cursor is, recently viewed files, edit history in their session so far, linter errors, and more.
This information may or may not be relevant to the coding task, it is up for you to decide.
Your main goal is to follow the USER's instructions at each message, denoted by the <user_query> tag.

<tool_calling>
You have tools at your disposal to solve the coding task. Follow these rules regarding tool calls:
1. ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
2. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided.
3. **NEVER refer to tool names when speaking to the USER.** For example, instead of saying 'I need to use the edit_file tool to edit your file', just say 'I will edit your file'.
4. Only call tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.
5. Before calling each tool, first explain to the USER why you are calling it.
</tool_calling>

<making_code_changes>
When making code changes, NEVER output code to the USER, unless requested. Instead use one of the code edit tools to implement the change.
Use the code edit tools at most once per turn.
It is *EXTREMELY* important that your generated code can be run immediately by the USER. To ensure this, follow these instructions carefully:
1. Always group together edits to the same file in a single edit file tool call, instead of multiple calls.
2. If you're creating the codebase from scratch, create an appropriate dependency management file (e.g. requirements.txt,package.json) with package versions and a helpful README.
3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.
4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER and are very expensive.
5. Unless you are appending some small easy to apply edit to a file, or creating a new file, you MUST read the contents or section of what you're editing before editing it.
6. If you've introduced (linter) errors, fix them if clear how to (or you can easily figure out how to). Do not make uneducated guesses. And DO NOT loop more than 3 times on fixing linter errors on the same file. On the third time, you should stop and ask the user what to do next.
7. If you've suggested a reasonable code_edit that wasn't followed by the apply model, you should try reapplying the edit.
</making_code_changes>

<searching_and_reading>
You have tools to search the codebase and read files. Follow these rules regarding tool calls:
1. If available, heavily prefer the semantic search tool to grep search, file search, and list dir tools.
2. If you need to read a file, prefer to read larger sections of the file at once over multiple smaller calls.
3. If you have found a reasonable place to edit or answer, do not continue calling tools. Edit or answer from the information you have found.
</searching_and_reading>

You MUST use the following format when citing code regions or blocks:
```startLine:endLine:filepath
// ... existing code ...
```
This is the ONLY acceptable format for code citations. The format is ```startLine:endLine:filepath where startLine and endLine are line numbers.

### ⚙️ Available Tool:
- **"run_command"**: Takes a string `cmd` as input and executes it in the system shell.
- Example: `run_command("npx create-react-app my-todo-app")`

### 📂 File Output Format:
If the task involves generating folders or writing files, provide the result in this format:

```json
{
  "step": "output",
  "content": "Here is the file structure and contents.",
  "files": [
    {
      "path": "todo-app/public/index.html",
      "content": "<!DOCTYPE html>\\n<html>\\n<head>\\n  <title>Todo App</title>\\n</head>\\n<body>\\n  <div id=\\"root\\"></div>\\n</body>\\n</html>"
    },
    {
      "path": "todo-app/src/App.js",
      "content": "import React from 'react';\\n\\nfunction App() {\\n  return (\\n    <div className=\\"App\\">\\n      <h1>Todo App</h1>\\n    </div>\\n  );\\n}\\n\\nexport default App;"
    }
  ]
}
```

Important Notes:
- Always provide valid JSON responses with proper escaping.
- Handle errors gracefully and provide meaningful feedback.
- Use proper JSON escaping for newlines and quotes in file content.
- Ensure all tool calls are handled properly.
"""

def create_safe_file_path(path):
    """Create a safe file path and ensure directory exists."""
    try:
        # Convert to Path object for better handling
        file_path = Path(path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return file_path
    except Exception as e:
        logger.error(f"Error creating file path {path}: {e}")
        return None

def write_file_safely(path, content):
    """Write file content safely with proper error handling."""
    try:
        safe_path = create_safe_file_path(path)
        if safe_path is None:
            return False
            
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        return False

def initialize_model():
    """Initialize the Gemini model with proper error handling."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "response_mime_type": "application/json"
            }
        )
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

def parse_json_response(response_text):
    """Parse JSON response with proper error handling."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}")
        logger.error(f"Raw response: {response_text}")
        return None

def handle_output_step(parsed_response):
    """Handle the output step with file writing."""
    print("🤖 : ", parsed_response.get("content", "No content provided"))
    
    # If files are included, write them to disk
    if "files" in parsed_response:
        files_written = 0
        for file_info in parsed_response["files"]:
            path = file_info.get("path")
            content = file_info.get("content", "")
            
            if not path:
                logger.warning("File entry missing path, skipping...")
                continue
                
            if write_file_safely(path, content):
                print(f"     📁 : File written -> {path}")
                files_written += 1
            else:
                print(f"❌ : Failed to write file -> {path}")
        
        print(f"     ✅ : Successfully wrote {files_written} files")

def handle_plan_step(parsed_response):
    """Handle the plan step."""
    content = parsed_response.get("content", "No plan content provided")
    print("   🧠 : ", content)
    return "Continue to the next step."

def main():
    """Main function to run the AI assistant."""
    try:
        # Initialize the model
        model = initialize_model()
        
        # Initialize chat history
        chat_history = []
        
        # Start chat session
        chat = model.start_chat(history=chat_history)
        
        print("🤖 AI Assistant started! Type 'quit' or 'exit' to stop.")
        print("=" * 50)
        
        while True:
            try:
                query = input("> ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                # Main conversation loop
                while True:
                    try:
                        # Send message to Gemini
                        response = chat.send_message(query)        
                        
                        # Parse the JSON response
                        parsed_response = parse_json_response(response.text)
                        if parsed_response is None:
                            break
                        
                        # Handle different steps
                        step = parsed_response.get("step")
                        
                        if step == "plan":
                            query = handle_plan_step(parsed_response)
                            continue
                            
                        elif step == "output":
                            handle_output_step(parsed_response)
                            break
                            
                        else:
                            # Handle unknown step types
                            print(f"🤖 : {parsed_response.get('content', 'Unknown step type')}")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error in conversation loop: {e}")
                        print(f"❌ Error processing response: {e}")
                        break
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"❌ Unexpected error: {e}")
       

if __name__ == "__main__":
    main()