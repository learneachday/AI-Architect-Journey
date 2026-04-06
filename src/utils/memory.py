from typing import List, Dict

conversation_history: List[Dict[str, str]] = []

def add_to_history(role: str, content: str):
    """Add message to conversation history"""
    conversation_history.append({"role": role, "content": content})

def get_history():
    """Return current conversation history"""
    return conversation_history.copy()

def clear_history():
    """Clear conversation history"""
    conversation_history.clear()
