from langgraph.graph import StateGraph,START,END 
from src.reportgenerator.states.states import State
from src.reportgenerator.nodes.route_node import route
from src.reportgenerator.nodes.abstract import generate_abstract
from src.reportgenerator.nodes.intro import generate_introduction
from src.reportgenerator.nodes.methodology import generate_methodology
from src.reportgenerator.nodes.proposed_method import generate_proposed_method
from src.reportgenerator.nodes.results import generate_results
from src.reportgenerator.nodes.references import generate_references
from src.reportgenerator.nodes.conclusion import generate_conclusion
from src.reportgenerator.nodes.final_report import final_report
from src.reportgenerator.nodes.orchestrator import orcehstrator
from src.reportgenerator.nodes.assign_workers import assign_workers
from src.reportgenerator.nodes.llm_call_parallel import llm_call
from src.reportgenerator.nodes.synthesizer import synthesizer



def Build_Graph():
    builder = StateGraph(State)
# builder.add_node("router",route)
    builder.add_node("abstract",generate_abstract)
    builder.add_node("introduction",generate_introduction)
    builder.add_node("methodology",generate_methodology)
    builder.add_node("proposed",generate_proposed_method)
    builder.add_node("results",generate_results)
    builder.add_node("references",generate_references)
    builder.add_node("conclusion",generate_conclusion)
    builder.add_node("final_report",final_report)
    builder.add_node("orchestrator",orcehstrator)
    builder.add_node("llm_call",llm_call)
    builder.add_node("synthesizer",synthesizer)
    
    
    builder.add_conditional_edges(
        START,
        route,
        {
            "User":"abstract",
            "Auto":"orchestrator"
        },
    )
    builder.add_edge("abstract","introduction")
    builder.add_edge("introduction","methodology")
    builder.add_edge("methodology","proposed")
    builder.add_edge("proposed","results")
    builder.add_edge("results","conclusion")
    builder.add_edge("conclusion","references")
    builder.add_edge("references","final_report")
    builder.add_edge("final_report",END)
    builder.add_conditional_edges(
        "orchestrator",
        assign_workers,
        ["llm_call"],
    )
    builder.add_edge("llm_call","synthesizer")
    builder.add_edge("synthesizer",END)
    return builder