import os
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import InferenceClient

# Updated RAG imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # ← Updated import

load_dotenv()

console = Console()

# ====================== LLM SETUP ======================
hf_token = os.getenv("HF_TOKEN")
if not hf_token or not hf_token.startswith("hf_"):
    console.print("[bold red]Error: HF_TOKEN not found in .env[/bold red]")
    exit(1)

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=hf_token,
)

# ====================== BASIC RAG SETUP ======================
console.print("[dim]Loading documents and building vector store...[/dim]")

# Load document
loader = TextLoader("data/ai_architect_notes.txt", encoding="utf-8")
documents = loader.load()

# Split document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

console.print(f"[dim]RAG system ready! Loaded {len(chunks)} document chunks.[/dim]")

# ====================== MEMORY ======================
from utils.memory import add_to_history, get_history, clear_history

def chat_with_llm(user_input: str):
    try:
        add_to_history("user", user_input)

        console.print("\n[bold green]AI is thinking...[/bold green]")

        # Retrieval: Find most relevant chunks
        relevant_docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Augmented prompt
        augmented_prompt = f"""Use only the following context to answer the question.
If the answer is not in the context, say "I don't have enough information from the provided documents."

Context:
{context}

Question: {user_input}

Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful AI Architect assistant."},
            {"role": "user", "content": augmented_prompt}
        ]

        # Streaming response
        console.print("\n[bold green]AI Response:[/bold green]")
        full_text = ""
        for token in client.chat.completions.create(
            messages=messages,
            max_tokens=700,
            temperature=0.6,
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
    console.print("[bold blue]=== AI Architect Journey - Day 6 (RAG Fixed) ===[/bold blue]")
    console.print("[dim]Basic RAG with updated HuggingFaceEmbeddings[/dim]\n")
    
    console.print("[yellow]Ask questions about the document[/yellow]:")
    console.print("1. What are the main responsibilities of an AI Architect?")
    console.print("2. What are the key trends in 2026?")
    console.print("3. What are the biggest challenges?")
    console.print("4. Clear (reset history)\n")

    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[cyan]Goodbye! See you for Day 7.[/cyan]")
            break
            
        if user_input.lower() == "clear":
            clear_history()
            console.print("[cyan]Conversation history cleared.[/cyan]")
            continue
            
        if user_input:
            chat_with_llm(user_input)
