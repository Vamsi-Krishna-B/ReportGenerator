# test_app.py
import pytest
# Import the function you want to test
from src.reportgeneratorapp.main import ReportGeneratorAPP

def test_reportgenerator_app_runs():
    """
    Test if the Report Generator app loads without raising an exception.
    """
    try:
        app = ReportGeneratorAPP()
    except Exception as e:
        pytest.fail(f"load_langgraph_agenticai_app() raised an exception: {e}")


if __name__ == "__main__":
    # Run the function manually if executing the test directly
    try:
        print("Running standalone test...")
        app = ReportGeneratorAPP()
        print("Success: App loaded without errors!")
    except Exception as e:
        print(f"Error: {e}")
