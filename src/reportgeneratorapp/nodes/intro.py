from src.reportgenerator.states.states import State
from langchain_core.messages import SystemMessage
from src.reportgenerator.tools.tools import TavilySearchContent
from src.reportgenerator.llms.GroqLLM import get_llm


def generate_introduction(state: State):
    """
    Generates the Introduction for the report based on the user input.
    """
    print("------------INTRODUCTION------------")
    extra_info = TavilySearchContent(state['user']['user_input']['about_problem'],top_k=15)
    prompt = [
        SystemMessage(
            content=f"""
                        You are a good researcher and can make standard reports according to the IEEE format. 
                        You are tasked to draft an Introduction for the report following the IEEE format based on the below content.
                        
                        Problem Statement:{state['user']['user_input']['about_problem']}
                        ----------------------------------------------------------------
                          \n\n
                        Also try to include the whole below given extra information in framing the introduction like stating about the problem and then 
                        include all of the data from extra information to give a story type large introduction discussing about all
                        of the methodologies used.
                        and finally add the proposed workflow and highlight how the current proposed method would be better and can improve results
                        Extra Information:
                        \n\n
                        {extra_info} 
                        \n\n
                        --------------------------------------------------------------------
                         Proposed Workflow : {state['user']['user_input']['proposed_workflow']}
                         
                         **DONOT MENTION ABOUT THE REFERENCES HERE**
                        """
        )
    ]
    llm = get_llm()
    introduction = llm.invoke(prompt, reasoning_format="hidden")
    user_data = state.get("user", {})
    # print(introduction.content)
    user_data["introduction"] = introduction.content
    return {"user":user_data}