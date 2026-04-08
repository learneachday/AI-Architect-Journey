import os
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import InferenceClient

# RAG imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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

# ====================== IMPROVED RAG SETUP ======================
console.print("[dim]Building improved RAG system with metadata...[/dim]")

# Load documents (supports both .txt and .pdf)
loader = TextLoader("data/ai_architect_notes.txt", encoding="utf-8")
# If you want to load PDF, use this instead:
# loader = PyPDFLoader("data/your_file.pdf")

documents = loader.load()

# Improved chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,          # Increased overlap for better context
    separators=["\n\n", "\n", ".", "!", "?"]
)

chunks = text_splitter.split_documents(documents)

# Add metadata to each chunk
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"] = "ai_architect_notes.txt"
    chunk.metadata["chunk_size"] = len(chunk.page_content)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

console.print(f"[dim]RAG Ready! Created {len(chunks)} chunks with metadata.[/dim]")

# ====================== MEMORY ======================
from utils.memory import add_to_history, get_history, clear_history

def chat_with_llm(user_input: str):
    try:
        add_to_history("user", user_input)

        console.print("\n[bold green]AI is thinking...[/bold green]")

        # === RETRIEVAL ===
        relevant_docs = vectorstore.similarity_search(user_input, k=4)

        # Log what was retrieved (very useful for debugging)
        console.print("[dim]Retrieved chunks:[/dim]")
        for i, doc in enumerate(relevant_docs):
            console.print(f"[dim]  Chunk {i+1} | Source: {doc.metadata['source']} | Score preview: ...[/dim]")

        context = "\n\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}" 
                              for doc in relevant_docs])

        # Better RAG Prompt (Production style)
        prompt = f"""You are an expert AI Architect assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, clearly say "I don't have sufficient information in the documents."

Context:
{context}

Question: {user_input}

Answer in a clear, professional manner. Mention the source when possible."""

        messages = [
            {"role": "system", "content": "You are a helpful, truthful AI Architect assistant."},
            {"role": "user", "content": prompt}
        ]

        # Streaming response
        console.print("\n[bold green]AI Response:[/bold green]")
        full_text = ""
        for token in client.chat.completions.create(
            messages=messages,
            max_tokens=800,
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
    console.print("[bold blue]=== AI Architect Journey - Day 7 ===[/bold blue]")
    console.print("[dim]Improved RAG: Metadata + Source Citations + Better Chunking + Retrieval Logging[/dim]\n")
    
    console.print("Test these questions:")
    console.print("1. What are the main responsibilities of an AI Architect?")
    console.print("2. Why is chunk overlap important?")
    console.print("3. What are the key trends in 2026?")
    console.print("4. Clear (reset history)\n")

    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[cyan]Goodbye! See you for Day 8.[/cyan]")
            break
            
        if user_input.lower() == "clear":
            clear_history()
            console.print("[cyan]Conversation history cleared.[/cyan]")
            continue
            
        if user_input:
            chat_with_llm(user_input)
