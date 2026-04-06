import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from rich.console import Console

load_dotenv()

console = Console()

# ====================== HUGGING FACE SETUP ======================
hf_token = os.getenv("HF_TOKEN")
if not hf_token or not hf_token.startswith("hf_"):
    console.print("[bold red]Error: HF_TOKEN not found or invalid in .env[/bold red]")
    console.print("→ Go to https://huggingface.co/settings/tokens and create a new 'Read' token")
    exit(1)

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=hf_token,
)

# ====================== IMPORT MEMORY ======================
from utils.memory import add_to_history, get_history, clear_history

def chat_with_llm(user_input: str):
    try:
        add_to_history("user", user_input)

        console.print("\n[bold green]AI is thinking...[/bold green]")

        # Prepare messages with history
        messages = [
            {"role": "system", "content": "You are a helpful AI Architect assistant. Be practical, concise, and focus on hands-on learning and architecture decisions."}
        ] + get_history()

        console.print("\n[bold green]AI Response:[/bold green]")
        full_text = ""

        # Streaming response
        for token in client.chat.completions.create(
            messages=messages,
            max_tokens=700,
            temperature=0.7,
            stream=True,
        ):
            content = token.choices[0].delta.content or ""
            console.print(content, end="")
            full_text += content

        console.print()

        add_to_history("assistant", full_text)
        return full_text

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return None


if __name__ == "__main__":
    console.print("[bold blue]=== AI Architect Journey - Day 5 (Stable HF Version) ===[/bold blue]")
    console.print("[dim]Using Llama-3.3-70B via Hugging Face (free tier)[/dim]\n")
    
    console.print("Test these:")
    console.print("1. What is an AI Architect in 2026?")
    console.print("2. Explain RAG simply")
    console.print("3. What are the main challenges in building production AI systems?")
    console.print("4. Clear (to reset history)\n")

    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[cyan]Goodbye! See you for Day 6.[/cyan]")
            break
            
        if user_input.lower() == "clear":
            clear_history()
            console.print("[cyan]Conversation history cleared.[/cyan]")
            continue
            
        if user_input:
            chat_with_llm(user_input)
