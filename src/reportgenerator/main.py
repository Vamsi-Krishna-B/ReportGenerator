import os
from src.reportgenerator.graphs.graph_builder import Build_Graph
from IPython.display import display, Image

class ReportGeneratorAPP:
    def __init__(self):
        os.environ["GROQ_API_KEY"] = ""
        os.environ["TAVILY_API_KEY"] = ""
        self.graph = Build_Graph().compile()
        self.final_report = '' 
    
    def save_as_md(self,file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.final_report)

    def ShowGraph(self):
        display(Image(self.graph.get_graph().draw_mermaid_png()))

    def generate_report(self, isUserInput, user_input=None,topic=None):
        final_report = ""
        if isUserInput:
            result = self.graph.invoke(
                {"is_userInput": isUserInput, "user": {"user_input": user_input}}
            )
            self.final_report = result['user']['final_report']
            return result['user']['final_report']
        else:
            result = self.graph.invoke(
                {
                    "is_userInput": False,
                    "auto": {
                        "topic":topic ,
                    },
                }
            ) 
            self.final_report = result['auto']['final_report']
            return result['auto']['final_report']