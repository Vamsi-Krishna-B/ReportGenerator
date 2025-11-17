from src.reportgeneratorapp.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgeneratorapp.llms.GroqLLM import get_llm

def generate_proposed_method(state:State):
    """
    Generates the proposed method Section of the report explaining how the workflow is.
    """
    print("------------PORPOSED METHOD-------------")
    user_proposed = state['user']['user_input']['proposed_workflow']
    prompt = [
        SystemMessage(
            content=f"""
                You are a good researcher and can make standard reports according to the IEEE format. 
               You are tasked to draft an Proposed Method for the report following the IEEE format based on the below content.
                You can understand the below given method workflow and explain and Enhance more about the method
                \n 
                {user_proposed}
                ** DONOT ADD REFERENCES HERE **
            """
        )
    ]
    llm = get_llm()
    result = llm.invoke(prompt,reasoning_format="hidden")
    user_data = state.get("user", {})
    user_data["proposed_method"] = result.content
    # print(result.content)
    return {"user":user_data}