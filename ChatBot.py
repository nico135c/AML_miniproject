import ollama
import tkinter as tk
from tkinter import ttk
import threading

class ChatBot:
    def __init__(self, name = "AML Bot", model="llama3.2", options={"temperature":0}):     
        # -- Model Variables --
        self.model = model
        self.options = options
        
        # -- Memory Variables --
        self.name = name
        self.chatlogs = "chat/chatlogs.txt"
        self.chat_summary = "chat/chat_summary.txt"

        # Create chatlogs and chat summary txt files if they do not exist
        open(self.chatlogs, "a").close()
        open(self.chat_summary, "a").close()
        
        self.max_summary_words = 250
        with open(self.chat_summary, "r", encoding="utf-8") as f:
            self.summary = f.read()  

    def _update_chatlogs(self, log):
        with open(self.chatlogs, "a", encoding="utf-8") as f:
            f.write(f"{log}\n")
    
    def _update_summary(self, u, a):
        sys = f"""
        You maintain a compact, factual summary of a long chat.
        Update the existing summary with NEW INFO from the latest exchange only.
        Keep important facts, user preferences, to-dos, and decisions.
        Remove info that is obsolete or contradicted.
        Target under {self.max_summary_words} words.

        Existing summary:
        {self.summary or "(none)"}

        Latest exchange:
        User: {u}
        Assistant: {a}

        Return ONLY the updated summary.
        """

        r = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": sys}],
            options=self.options
        )
        self.summary = r["message"]["content"]
    
    def _save_summary(self):
        with open(self.chat_summary, "w", encoding="utf-8") as f:
            f.write(self.summary)

    def generate_init_msg(self):
        sys = f"""
        Generate a friendly opening message for a user starting a chat session.  
        You are a helpful assistant named {self.name}, so begin by introducing yourself clearly.  
        
        After the introduction, include a brief recap (about 20–50 words) of what you and the user have previously discussed, based on the conversation summary provided.  
        The recap should feel natural and smoothly integrated into your greeting.  
        
        If useful, you may weave in factual details from the summary, but never invent or use placeholder data.
        Keep the tone warm, professional, and concise.  
        
        Conversation summary (up to ~250 words):  
        {self.summary or "(none)"}
        """

        r = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": sys}],
            options=self.options
        )
        msg = r["message"]["content"]
        self._update_chatlogs(f"(Assistant): {msg}")
        return msg
    
    def prompt(self, text):
        self._update_chatlogs(f"(User): {text}")
        sys = f"""
            You are a helpful assistant with the name {self.name}

            Use these persistent notes if relevant:

            PERSISTENT NOTES:
            {self.summary}
            """
        
        r = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": text},
            ],
            options=self.options
        )
        response = r["message"]["content"]
        self._update_chatlogs(f"(Assistant): {response}")

        self._update_summary(text, response)

        return response
    
    def exit(self):
        self._save_summary()
        return

class ChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.chatbot = ChatBot()
        self.title(f"{self.chatbot.name} UI")
        self.geometry("500x400")

        self._alive = True
        self._inflight = 0
        self._typing_job = None
        self._dot_count = 0

        # --- Chat history ---
        self.chat_text = tk.Text(self, state="disabled", wrap="word")
        self.chat_text.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.chat_text, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self.chat_text.tag_configure("user_name", foreground="blue", font=("", 10, "bold"))
        self.chat_text.tag_configure("bot_name", foreground="green", font=("", 10, "bold"))

        # --- Input area ---
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        self.entry = ttk.Entry(bottom_frame)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.entry.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(bottom_frame, text="Send", command=self._on_send)
        self.send_btn.pack(side="right")

        # Status label
        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", padx=6, pady=(0,6))

        # Init message in background
        self._start_typing()
        threading.Thread(target=self._init_worker, daemon=True).start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI helpers ----------------
    def append_message(self, role, text):
        self.chat_text.configure(state="normal")
        if role.lower() == "you":
            self.chat_text.insert("end", f"{role}: ", "user_name")
        elif role.lower() == self.chatbot.name.lower():
            self.chat_text.insert("end", f"{role}: ", "bot_name")
        else:
            self.chat_text.insert("end", f"{role}: ")
        self.chat_text.insert("end", text + "\n")
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    # Typing indicator with cycling dots
    def _start_typing(self):
        self._dot_count = 0
        def animate():
            dots = "." * (self._dot_count % 4)
            self.status_var.set(f"{self.chatbot.name} is typing{dots}")
            self._dot_count += 1
            self._typing_job = self.after(500, animate)  # repeat every 0.5s
        animate()

    def _stop_typing(self):
        if self._typing_job:
            self.after_cancel(self._typing_job)
            self._typing_job = None
        self.status_var.set("")

    def _lock_input(self, locked: bool):
        state = "disabled" if locked else "normal"
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)

    # ---------------- Event handlers ----------------
    def _on_send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.append_message("You", text)

        self._inflight += 1
        self._lock_input(True)
        self._start_typing()
        threading.Thread(target=self._prompt_worker, args=(text,), daemon=True).start()

    # ---------------- Worker threads ----------------
    def _init_worker(self):
        try:
            msg = self.chatbot.generate_init_msg()
        except Exception as e:
            msg = f"(error during init) {e}"
        if self._alive:
            self.after(0, lambda m=msg: (self.append_message(self.chatbot.name, m), self._stop_typing()))

    def _prompt_worker(self, text):
        try:
            reply = self.chatbot.prompt(text)
        except Exception as e:
            reply = f"(error) {e}"

        if self._alive:
            def done():
                self.append_message(self.chatbot.name, reply)
                self._inflight -= 1
                if self._inflight <= 0:
                    self._lock_input(False)
                    self._stop_typing()
            self.after(0, done)

    def on_close(self):
        self._alive = False
        self._stop_typing()
        self.chatbot.exit()
        self.destroy()
