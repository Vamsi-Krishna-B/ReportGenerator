from src.reportgenerator.states.states import State

def route(state:State):
    """
    Routes the graph flow based on the decision taken by the user.
    """
    print("------------ROUTING-------------")
    if state['is_userInput']:
        return "User"
    else:
        return "Auto"