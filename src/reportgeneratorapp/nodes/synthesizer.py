from src.reportgenerator.states.states import State

def synthesizer(state:State):
    """Synthesize full report from sections"""
    print(state['auto'].keys())
    completed_sections = state["completed_sections"]
    
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"auto":{"final_report":completed_report_sections}}