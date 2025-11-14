import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
from src.reportgenerator.main import ReportGeneratorAPP
from IPython.display import Markdown,display

report_app = ReportGeneratorAPP()
report_app.ShowGraph()
# result = app.generate_report(isUserInput=False,topic="Agentic AI vs AI Agents")
user_input = {
        "title": "Laryngeal Cancer Detection Using Deep CNN and Feature Fusion",
        "about_problem": "Recent researches done on Laryngeal Cancer detection",
        
        "methods_used": "ResNet152V2 CNN, SFTA texture analysis, feature fusion, Linear Discriminant Analysis, Kernel SVM, K-fold cross-validation.",
        
        "proposed_workflow": "Collect and preprocess laryngeal images. Extract deep features with ResNet152V2 and texture features with SFTA. Fuse features, reduce dimensionality with LDA, and classify using Kernel SVM. Evaluate with K-fold cross-validation.",
        
        "results": "The model achieved 99.89% training and 99.85% testing accuracy, demonstrating strong generalization and robustness for automated laryngeal cancer detection."
    }
result = report_app.generate_report(isUserInput=True,user_input=user_input)
report_app.save_as_md("laryngeal.md")