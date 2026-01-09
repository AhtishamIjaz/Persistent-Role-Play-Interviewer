import sqlite3
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Load API Key from .env
load_dotenv()

# 1. Define the State
class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    resume_context: str
    interview_stage: str 
    stress_level: int     

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# 3. Define the Node: Interviewer Logic
def interviewer_node(state: InterviewState):
    messages = state['messages']
    resume = state.get('resume_context', "No resume provided.")
    stage = state.get('interview_stage', "Introduction")
    stress = state.get('stress_level', 1)

    # Persona Switching Logic
    if stress <= 2:
        persona = "a Friendly HR Recruiter"
    elif 3 <= stress <= 4:
        persona = "a Senior Software Engineer"
    else:
        persona = "a strict Technical Lead"

    system_instructions = f"""
    You are {persona}. Stage: {stage}. Resume: {resume}.
    Task: Conduct a realistic interview. Ask ONE question. 
    If the user answers well, get slightly more technical.
    """

    prompt = [SystemMessage(content=system_instructions)] + messages
    response = llm.invoke(prompt)

    # State Update Logic
    new_stage = "Technical" if len(messages) > 4 else "Introduction"
    
    return {
        "messages": [response],
        "interview_stage": new_stage,
        "stress_level": stress + 1 if len(messages) % 3 == 0 else stress
    }

# 4. Build Graph
workflow = StateGraph(InterviewState)
workflow.add_node("interviewer", interviewer_node)
workflow.add_edge(START, "interviewer")
workflow.add_edge("interviewer", END)

# 5. SQLite Persistence
DB_PATH = "interview_history.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)

# 6. Helper to Fetch History for Sidebar
def get_all_threads():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []