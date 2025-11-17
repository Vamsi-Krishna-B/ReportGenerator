from src.reportgeneratorapp.states.states import State
from src.reportgeneratorapp.tools.tools import WikiSearchContent

def generate_methodology(state:State):
    """
     Generates the Methodology Section of the report explaining and highlighting about the methodologies used in the proposed work.
    """
    print("------------METHODOLOGY-------------")
    methods = state['user']['user_input']['methods_used']
    methods = methods.split(",")
    methodology = ' **Methodology** \n'
    for method in methods:
        methodology+= WikiSearchContent(method)
    # print(methodology)
    user_data = state.get("user", {})
    user_data["methodology"] = methodology
    return {"user":user_data}