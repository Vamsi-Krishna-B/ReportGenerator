from flask import Flask, render_template, request, redirect, url_for
from src.reportgeneratorapp.main import ReportGeneratorAPP
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
app = Flask(__name__)
report_app = ReportGeneratorAPP()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    user_input = request.form.get('user_input')

    if user_input == "Yes":
        return redirect(url_for('input_page'))
    else:
        return redirect(url_for('auto_generate_page'))
    
@app.route('/auto_generate_page')
def auto_generate_page():
    return render_template('auto_generate_page.html')

@app.route('/input_page')
def input_page():
    return render_template('input_page.html')

@app.route('/input_generate', methods=['POST'])
def input_generate():
    # Collect all fields from the form
    title = request.form.get('title')
    about_problem = request.form.get('about_problem')
    methods_used = request.form.get('methods_used')
    proposed_workflow = request.form.get('proposed_workflow')
    results = request.form.get('results')
    user_input = {"title":title,
                  "about_problem":about_problem,
                  "methods_used":methods_used,
                  "proposed_workflow":proposed_workflow,
                  "results":results}
    result = report_app.generate_report(isUserInput=True,user_input=user_input)
    # You can process or pass these to your model here
    return result

@app.route('/auto_generate', methods=['POST'])
def auto_generate():
    topic = request.form.get('topic')
    result = report_app.generate_report(isUserInput=False,topic=topic)
    return result

if __name__ == '__main__':
    app.run()
