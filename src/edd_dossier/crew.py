import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class EddDossierCrew:
    """Four specialists producing one dossier: media, ownership, screening, and the
    writer who assembles them. Sequential, because the dossier can only be written
    once the other three have reported."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Gemini rather than the OpenAI default: the Vertex agent in the Decision stage
    # runs on the same family, so the whole case reasons on one model.
    LLM_MODEL = "gemini/gemini-2.5-flash"

    def _llm(self) -> LLM:
        # Pass the key explicitly rather than trusting provider env lookup. When it
        # is missing the platform only surfaces a generic "Authentication required",
        # which says nothing about what the container actually got -- so name the
        # variables we looked for, and what env keys are present, without leaking values.
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            visible = sorted(k for k in os.environ if any(
                t in k.upper() for t in ("GEMINI", "GOOGLE", "API", "KEY", "LLM")))
            raise RuntimeError(
                "No Gemini key in the container. Looked for GEMINI_API_KEY and "
                "GOOGLE_API_KEY. Env keys that look related: "
                + (", ".join(visible) if visible else "(none)")
                + f" | total env vars: {len(os.environ)}")
        return LLM(model=self.LLM_MODEL, temperature=0.2, api_key=key)

    @agent
    def media_researcher(self) -> Agent:
        return Agent(config=self.agents_config["media_researcher"], llm=self._llm(), verbose=True)

    @agent
    def registry_analyst(self) -> Agent:
        return Agent(config=self.agents_config["registry_analyst"], llm=self._llm(), verbose=True)

    @agent
    def sanctions_contextualiser(self) -> Agent:
        return Agent(config=self.agents_config["sanctions_contextualiser"], llm=self._llm(), verbose=True)

    @agent
    def edd_writer(self) -> Agent:
        return Agent(config=self.agents_config["edd_writer"], llm=self._llm(), verbose=True)

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
