from src.reportgeneratorapp.states.states import State
from langchain_core.messages import SystemMessage,HumanMessage
from src.reportgeneratorapp.llms.GroqLLM import get_llm

def llm_call(state:State):
    """Worker writes a section of the report"""
    llm = get_llm()
    section = llm.invoke(
        [
            SystemMessage(
                        content=f"Write a report section following the provided name and description. Include no preamble for each section.Used markdown formatting"
            ),
            HumanMessage(
                content=f"here is the section name : {state['auto']['section'].title} and description: {state['auto']['section'].description}"
            )
        ],reasoning_format="hidden"
    )
    return {"completed_sections":[section.content]}