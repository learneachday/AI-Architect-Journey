import os
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import InferenceClient

# RAG imports
from langchain_community.document_loaders import TextLoader
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

# ====================== IMPROVED RAG WITH RERANKING ======================
console.print("[dim]Building RAG system with reranking...[/dim]")

loader = TextLoader("data/ai_architect_notes.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?"]
)

chunks = text_splitter.split_documents(documents)

# Add metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["source"] = "ai_architect_notes.txt"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

console.print(f"[dim]RAG Ready! {len(chunks)} chunks created.[/dim]")

# ====================== MEMORY ======================
from utils.memory import add_to_history, get_history, clear_history

def chat_with_llm(user_input: str):
    try:
        add_to_history("user", user_input)

        console.print("\n[bold green]AI is thinking...[/bold green]")

        # Step 1: Initial Retrieval (Similarity Search)
        initial_docs = vectorstore.similarity_search(user_input, k=6)

        # Step 2: Reranking (using a cross-encoder reranker)
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        
        # Prepare pairs for reranker
        pairs = [[user_input, doc.page_content] for doc in initial_docs]
        scores = reranker.compute_score(pairs, normalize=True)
        
        # Sort documents by reranker score (higher = better)
        reranked_docs = [doc for _, doc in sorted(zip(scores, initial_docs), reverse=True)]
        top_docs = reranked_docs[:4]   # Take top 4 after reranking

        # Show what was finally selected
        console.print("[dim]Top retrieved & reranked chunks:[/dim]")
        for i, doc in enumerate(top_docs):
            console.print(f"[dim]  {i+1}. Chunk {doc.metadata['chunk_id']} (Score: high)[/dim]")

        context = "\n\n".join([
            f"Source: {doc.metadata['source']} (Chunk {doc.metadata['chunk_id']})\n{doc.page_content}" 
            for doc in top_docs
        ])

        # Strong RAG Prompt
        prompt = f"""Answer the question using ONLY the provided context below. 
Be accurate and concise. If the answer is not in the context, say "I don't have enough information from the documents."

Context:
{context}

Question: {user_input}

Answer:"""

        messages = [
            {"role": "system", "content": "You are an expert AI Architect assistant. Always cite sources when possible."},
            {"role": "user", "content": prompt}
        ]

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
    console.print("[bold blue]=== AI Architect Journey - Day 8 ===[/bold blue]")
    console.print("[dim]Advanced RAG: Reranking + Better Citations + Retrieval Logging[/dim]\n")
    
    console.print("Test these:")
    console.print("1. Why is chunk overlap important in RAG?")
    console.print("2. What are the key trends in 2026?")
    console.print("3. What are the main responsibilities of an AI Architect?")
    console.print("4. Clear (reset history)\n")

    while True:
        user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[cyan]Goodbye! See you for Day 9.[/cyan]")
            break
            
        if user_input.lower() == "clear":
            clear_history()
            console.print("[cyan]Conversation history cleared.[/cyan]")
            continue
            
        if user_input:
            chat_with_llm(user_input)
