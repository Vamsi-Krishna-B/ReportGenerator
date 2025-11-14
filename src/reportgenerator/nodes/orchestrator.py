from src.reportgenerator.states.states import State,Sections
from langchain_core.messages import SystemMessage,HumanMessage
from src.reportgenerator.llms.GroqLLM import get_llm


def orcehstrator(state:State):
    """ Orchestrtor that generates plan for the report"""
    print("In orchestrator")
    auto_planner = get_llm().with_structured_output(Sections)
    report_sections = auto_planner.invoke(
        [
            SystemMessage(content="You are a world class research assistant,and you are great at creating outlines for reports"),
            HumanMessage(content=f"Create a detailed outline for a report on the topic:{state['auto']['topic']}.List at least 5 sections with name and description"),
        ],reasoning_format="hidden"
    )
    print("In orchestrator after llm")
    # print("Report Sections:",report_sections)

    return {"auto":{"sections":report_sections.sections}}