import sys
from crew import BloodHoundCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        "path_to_blood_report": "./data/sample_blood_report.pdf",
    }
    print(BloodHoundCrew().crew().kickoff())
