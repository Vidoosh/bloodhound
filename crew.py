from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from crewai_tools import FileReadTool, SerperDevTool, WebsiteSearchTool, DirectoryReadTool
import openai
import logging


@CrewBase
class BloodHoundCrew:
    """BloodHound crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    file_tool = FileReadTool()
    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()

    @agent
    def analyzer(self, file_path="./data/sample_blood_report.pdf") -> Agent:
        return Agent(
            config=self.agents_config["analyzer"],
            tools=[self.file_tool],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[self.search_tool, self.web_rag_tool],
            verbose=True,
            allow_delegation=False,
            # llm=self.llm
        )
    
    @agent
    def filter(self) -> Agent:
        return Agent(
            config=self.agents_config["filter"],
            tools=[self.search_tool, self.web_rag_tool],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["advisor"],
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def analyze_health_report(self) -> Task:
        print("analyzing...")
        return Task(config=self.tasks_config["analyze_health_report"], agent=self.analyzer)

    @task
    def find_relevant_articles(self) -> Task:
        print("searching...")
        return Task(config=self.tasks_config["find_relevant_articles"], agent=self.researcher)

    @task
    def find_relevant_articles(self) -> Task:
        print("filtering...")
        return Task(config=self.tasks_config["filter_articles"], agent=self.filter)
    
    @task
    def find_relevant_articles(self) -> Task:
        print("advicing...")
        return Task(config=self.tasks_config["health_recommendations"], agent=self.advisor)

    @crew
    def crew(self) -> Crew:
        """Creates the BloodHound crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2,
            full_output=True
        )
