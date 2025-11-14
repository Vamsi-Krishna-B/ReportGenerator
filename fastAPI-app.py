import uvicorn
import warnings
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from src.reportgenerator.main import ReportGeneratorAPP
from fastapi.staticfiles import StaticFiles
report_app = ReportGeneratorAPP()
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
async def index(request:Request):
   return templates.TemplateResponse(
        "index.html",  # name of the template
        {"request": request, "title": "Welcome Page", "name": "Vamsi"}
    )

@app.post("/generate_report")
async def generate_report(user_input: str = Form(...)):
    if user_input == "Yes":
        return RedirectResponse(url="/input_page", status_code=303)
    else:
        return RedirectResponse(url="/auto_generate_page", status_code=303)

@app.get("/auto_generate_page", response_class=HTMLResponse)
async def auto_generate_page(request: Request):
    return templates.TemplateResponse("auto_generate_page.html", {"request": request})


@app.get("/input_page", response_class=HTMLResponse)
async def input_page(request: Request):
    return templates.TemplateResponse("input_page.html", {"request": request})


# ---------- POST ROUTES ----------

@app.post("/input_generate")
async def input_generate(
    title: str = Form(...),
    about_problem: str = Form(...),
    methods_used: str = Form(...),
    proposed_workflow: str = Form(...),
    results: str = Form(...)
):
    user_input = {
        "title": title,
        "about_problem": about_problem,
        "methods_used": methods_used,
        "proposed_workflow": proposed_workflow,
        "results": results
    }

    # Call your report generation function
    result = report_app.generate_report(isUserInput=True, user_input=user_input)
    return result


@app.post("/auto_generate")
async def auto_generate(topic: str = Form(...)):
    result = report_app.generate_report(isUserInput=False, topic=topic)
    return result


# ---------- REDIRECT LOGIC ----------

@app.post("/generate_report")
async def generate_report(user_input: str = Form(...)):
    if user_input == "Yes":
        return RedirectResponse(url="/input_page", status_code=303)
    else:
        return RedirectResponse(url="/auto_generate_page", status_code=303)


if __name__ == "__main__":
    uvicorn.run("fastAPI-app:app", host="0.0.0.0", port=8000,reload=True)