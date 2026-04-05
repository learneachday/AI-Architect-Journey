import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from rich.console import Console
from rich.markdown import Markdown

# Import our new modules
from src.tools.calculator import calculator
from src.utils.memory import add_to_history, get_history, clear_history

load_dotenv()

console = Console()

hf_token = os.getenv("HF_TOKEN")
if not hf_token or not hf_token.startswith("hf_"):
    console.print("[bold red]Error: Invalid or missing HF_TOKEN in .env[/bold red]")
    exit(1)

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=hf_token,
)

# ====================== TOOLS DEFINITION ======================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Use this tool ONLY when the user asks for mathematical calculations, percentages, or simple math problems. Examples: 'What is 15*23?', 'Calculate 20% of 850'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to calculate (e.g. '2+2', '15*23', '100/4 + 20')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def execute_tool(tool_call):
    """Execute the requested tool and return result"""
    function_name = tool_call.function.name
    arguments = tool_call.function.arguments  # This is a JSON string

    if function_name == "calculator":
        import json
        args = json.loads(arguments)
        return calculator(args.get("expression", ""))
    
    return "Unknown tool"

def chat_with_llm(user_input: str):
    try:
        add_to_history("user", user_input)

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful AI Architect assistant. Be practical and concise."}
        ] + get_history()

        console.print("\n[bold green]AI is thinking...[/bold green]")

        # First call - let model decide if it needs a tool
        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=600,
            temperature=0.7,
            stream=False   # We use non-stream for tool calls to make parsing easier
        )

        message = response.choices[0].message

        # Check if model wants to call a tool
        if message.tool_calls:
            console.print("[yellow]Tool is being used...[/yellow]")
            
            for tool_call in message.tool_calls:
                result = execute_tool(tool_call)
                console.print(f"[dim]Tool result: {result}[/dim]")
                
                # Add tool result back to conversation
                add_to_history("assistant", f"[Tool used: {tool_call.function.name}] Result: {result}")
                
                # Second call to get final natural language response
                messages = [
                    {"role": "system", "content": "You are a helpful AI Architect assistant."}
                ] + get_history()
                
                final_response = client.chat.completions.create(
                    messages=messages,
                    max_tokens=600,
                    temperature=0.7,
                    stream=True
                )
                
                console.print("\n[bold green]AI Response:[/bold green]")
                full_text = ""
                for chunk in final_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        console.print(content, end="")
                        full_text += content
                console.print()
                
                add_to_history("assistant", full_text)
                return full_text

        else:
            # No tool needed - normal response with streaming
            console.print("\n[bold green]AI Response:[/bold green]")
            full_text = ""
            stream = client.chat.completions.create(
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    console.print(content, end="")
                    full_text += content
            console.print()
            
            add_to_history("assistant", full_text)
            return full_text

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return None

if __name__ == "__main__":
    console.print("[bold blue]=== AI Architect Journey - Day 3 ===[/bold blue]")
    console.print("[dim]Improved: Automatic Tool Execution + Better Memory[/dim]\n")
    console.print("[yellow]Try: What is 245 * 17?   or   Calculate 20% of 850[/yellow]\n")

    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        if user_input.lower() in ["exit", "quit", "bye", "clear"]:
            if user_input.lower() == "clear":
                clear_history()
                console.print("[cyan]Conversation history cleared.[/cyan]")
            else:
                console.print("[cyan]Goodbye! See you for Day 4.[/cyan]")
                break
        if user_input:
            chat_with_llm(user_input)

