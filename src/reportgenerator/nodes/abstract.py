from src.reportgenerator.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgenerator.llms.GroqLLM import get_llm

def generate_abstract(state:State):
    """
    Generates the abstract for the report based on the user input.
    """
    print("------------ABSRACT-------------")
    prompt = [
        SystemMessage(
            content=f"""
                        You are a good researcher and can make standard reports according to the IEEE format. 
                        You are tasked to make an abstract for the report following the IEEE format based on the below content.
                        Also use your own knowledge about neatly presenting the abstract
                        Title : {state['user']['user_input']['title']}
                        Problem Statement: {state['user']['user_input']['about_problem']}
                        Proposed Workflow : {state['user']['user_input']['proposed_workflow']}
                        Results : {state['user']['user_input']['results']}
                        ---------
                        """
        )
    ]
    llm = get_llm()
    abstract = llm.invoke(prompt,reasoning_format="hidden")
    user_data = state.get("user", {})
    user_data["abstract"] = abstract.content
    return {"user":user_data}    