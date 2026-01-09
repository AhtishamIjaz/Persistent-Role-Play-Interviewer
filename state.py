from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    messages: Annotated[List, add_messages]
    resume_context: str
    interview_stage: str  # Intro -> Technical -> Behavioral
    stress_level: int     # 1 (Friendly) to 5 (High Pressure)
    feedback: List[str]   # Specific mistakes to remember for next time