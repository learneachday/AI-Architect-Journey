import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="AI Architect Agent", page_icon="🚀", layout="wide")

st.title("🚀 AI Architect Agent")
st.caption("Your personal AI Architecture assistant with RAG + Tools + Memory")

# Persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load history from file if exists
HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE) and len(st.session_state.messages) == 0:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    except:
        pass

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything about AI Architecture..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from agent.react_agent import ReActAgent
            from rag.vectorstore import get_vectorstore
            from tools.calculator import calculator
            from tools.web_search import web_search

            vectorstore = get_vectorstore()
            agent = ReActAgent(vectorstore=vectorstore, calculator=calculator,web_search=web_search)

            response = agent.chat(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Save to file
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, indent=2, ensure_ascii=False)

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()
    
    st.header("Loaded Documents")
    try:
        from rag.loader import list_loaded_documents
        for doc in list_loaded_documents():
            st.write(f"📄 {doc}")
    except:
        st.write("No documents loaded")

    st.caption("Day 20 - Persistent Memory")
