def web_search(query: str) -> str:
    """Simple web search simulator tool (for learning)"""
    # In real projects, we would call an actual search API here
    # For now, we simulate useful responses
    query_lower = query.lower()
    
    if "weather" in query_lower:
        return "Current weather in Delhi: 28°C, partly cloudy. (simulated)"
    elif "news" in query_lower or "latest" in query_lower:
        return "Latest AI news: Agentic AI and GraphRAG are trending in 2026. (simulated)"
    elif "rag" in query_lower:
        return "RAG stands for Retrieval Augmented Generation. It helps LLMs answer using your own documents."
    elif "ai architect" in query_lower:
        return "An AI Architect designs end-to-end AI systems, chooses models, tools, and ensures scalability & governance."
    else:
        return f"Search result for '{query}': This is a simulated response. In production, we would call Tavily, Serper, or Brave Search API."
