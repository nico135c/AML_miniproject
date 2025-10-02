# AML Bot  

AML Bot is a simple desktop chatbot built with **Python**, **Tkinter**, and **Ollama**.  
It provides a local chat interface to an LLM model (default: `llama3.2`) and keeps track of conversations over time.  

## Features  
- Tkinter GUI with scrollable chat window and input box  
- Responsive chat thanks to threading  
- Uses [Ollama](https://ollama.com) to run local language models  
- Saves full chat history to `chat/chatlogs.txt`  
- Maintains a rolling summary in `chat/chat_summary.txt` for persistent memory  
- Starts each session with a friendly greeting that recalls past conversations  

## How to run  
1. Install [Ollama](https://ollama.com) and pull the model you want (default: `ollama pull llama3.2`).  
2. Clone this repo and create a Python virtual environment.  
3. Run the app:  
   ```bash
   python main.py
