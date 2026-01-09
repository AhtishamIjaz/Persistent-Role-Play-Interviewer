import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph_builder import graph, get_all_threads
from utils import extract_text_from_pdf

st.set_page_config(page_title="CampusX Interview Agent", layout="wide")

# Initialize Session States
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# --- SIDEBAR: HISTORY & UPLOAD ---
with st.sidebar:
    st.title("Settings & History")
    
    if st.button("➕ New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Previous Chats")
    threads = get_all_threads()
    for t_id in threads:
        if st.button(f"Session {t_id[:8]}...", key=t_id):
            st.session_state.thread_id = t_id
            # Load messages from SQLite for this thread
            state_data = graph.get_state({"configurable": {"thread_id": t_id}})
            st.session_state.messages = state_data.values.get("messages", [])
            st.rerun()

    st.divider()
    st.subheader("Resume Upload")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume Parsed!")

# --- MAIN CHAT INTERFACE ---
st.title("🤖 AI Role-Play Interviewer")
st.info(f"Active Thread: {st.session_state.thread_id}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# Chat Input & Streaming
if prompt := st.chat_input("Start speaking..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "resume_context": st.session_state.resume_text
        }

        # Streaming loop
        for event in graph.stream(inputs, config=config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    full_response = last_msg.content
                    placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.messages.append(AIMessage(content=full_response))