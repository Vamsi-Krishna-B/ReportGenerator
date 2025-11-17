from src.reportgeneratorapp.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgeneratorapp.llms.GroqLLM import get_llm


def generate_results(state:State):
    """
    Generates the result section of the report explaining
    """
    print("------------RESULTS-------------")
    user_results = state['user']['user_input']['results']
    prompt = [
        SystemMessage(
            content=f"""
                You are a good researcher and can make standard reports according to the IEEE format. 
               You are tasked to draft an  Result for the report following the IEEE format based on the below content.
                Frame the results in well mannered format.
                \n 
                {user_results}
            """
        )
    ]
    llm = get_llm()
    result = llm.invoke(prompt,reasoning_format="hidden")
    user_data = state.get("user", {})
    user_data["results"] = result.content
    # print(result.content)
    return {"user":user_data}