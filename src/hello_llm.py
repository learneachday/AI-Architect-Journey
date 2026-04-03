import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from rich.console import Console

load_dotenv()

console = Console()

# Free Inference API - fast enough for learning
client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",  # or "Qwen/Qwen2.5-72B-Instruct" etc.
    token=os.getenv("HF_TOKEN")
)

def chat_with_llm(user_input: str):
    try:
        console.print("\n[bold green]AI Response:[/bold green]")
        full_response = ""
        for token in client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            max_tokens=500,
            temperature=0.7,
            stream=True,
        ):
            content = token.choices[0].delta.content or ""
            console.print(content, end="")
            full_response += content
        console.print()  # new line
        return full_response
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None

if __name__ == "__main__":
    console.print("[bold blue]=== AI Architect Journey - Day 1 CLI Chat (Hugging Face) ===[/bold blue]")
    console.print("[dim]Using Llama-3.3-70B via HF Inference[/dim]")
    
    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[cyan]Goodbye! See you tomorrow for Day 2.[/cyan]")
            break
        chat_with_llm(user_input)

