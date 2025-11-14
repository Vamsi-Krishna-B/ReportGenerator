from src.reportgenerator.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgenerator.llms.GroqLLM import get_llm


def generate_conclusion(state:State):
    """
    Generates the conclusion section of the report
    """
    print("------------CONCLUSION-------------")
    prompt = [
        SystemMessage(
            content=f"""
            You are a good researcher and can make standard reports according to the IEEE format. 
               You are tasked to draft the conclusion section for the report following the IEEE format based on the below content.
               Elaborate and Include how in future new methods can be added to this work.
               \n\n
               {state['user']['abstract']}
            """
        )
    ]
    llm = get_llm()
    result = llm.invoke(prompt,reasoning_format="hidden")
    user_data = state.get("user", {})
    user_data["conclusion"] = result.content
    # print(result.content)
    return {"user":user_data}