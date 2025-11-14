from src.reportgenerator.states.states import State


def final_report(state:State):
    """
    Combines the all generated section's content
    """
    combined_sections = "\n\n".join(
    str(value) for key, value in state['user'].items() if key != "user_input"
)   
    user_data = state.get("user", {})
    user_data["final_report"] = combined_sections
    # print(result.content)
    return {"user":user_data}