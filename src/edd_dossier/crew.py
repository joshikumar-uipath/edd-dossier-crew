from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EddDossierCrew:
    """Four specialists producing one dossier: media, ownership, screening, and the
    writer who assembles them. Sequential, because the dossier can only be written
    once the other three have reported."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def media_researcher(self) -> Agent:
        return Agent(config=self.agents_config["media_researcher"], verbose=True)

    @agent
    def registry_analyst(self) -> Agent:
        return Agent(config=self.agents_config["registry_analyst"], verbose=True)

    @agent
    def sanctions_contextualiser(self) -> Agent:
        return Agent(config=self.agents_config["sanctions_contextualiser"], verbose=True)

    @agent
    def edd_writer(self) -> Agent:
        return Agent(config=self.agents_config["edd_writer"], verbose=True)

    @task
    def research_media(self) -> Task:
        return Task(config=self.tasks_config["research_media"])

    @task
    def resolve_ownership(self) -> Task:
        return Task(config=self.tasks_config["resolve_ownership"])

    @task
    def contextualise_screening(self) -> Task:
        return Task(config=self.tasks_config["contextualise_screening"])

    @task
    def write_dossier(self) -> Task:
        return Task(config=self.tasks_config["write_dossier"])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks,
                    process=Process.sequential, verbose=True)
