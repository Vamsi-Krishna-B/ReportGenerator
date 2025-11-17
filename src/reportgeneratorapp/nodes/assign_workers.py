from src.reportgeneratorapp.states.states import State
from langgraph.types import Send 

def assign_workers(state:State):
    """Assign workers to each section of the report"""
    return [Send("llm_call",{"auto":{"section":s}}) for s in state['auto']['sections']]