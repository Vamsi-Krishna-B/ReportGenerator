from src.reportgenerator.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgenerator.llms.GroqLLM import get_llm


def generate_references(state:State):
    """
    Generates the References section of the report.
    """
    print("------------REFERENCES-------------")
    methods = state['user']['user_input']['methods_used']
    prompt = [
        SystemMessage(
            content=f"""
            You are a good researcher and can make standard reports according to the IEEE format. 
               You are tasked to draft an  References for the report following the IEEE format based on the below content.
               Extract the two refernce per method information as per IEEE format from below methods.
               {methods}
            """
        )
    ]
    llm = get_llm()
    result = llm.invoke(prompt,reasoning_format="hidden")
    user_data = state.get("user", {})
    user_data["references"] = result.content
    # print(result.content)
    return {"user":user_data}