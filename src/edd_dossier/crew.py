import hashlib
import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, before_kickoff, crew, task

# Both names the Gemini provider consults, in the order it consults them.
KEY_ENV_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")

# Accepted spellings for the key when it arrives with the request. The Inputs tab
# does not declare it, so a caller may reasonably guess at any of these.
KEY_INPUTS = ("gemini_api_key", "GEMINI_API_KEY", "google_api_key", "GOOGLE_API_KEY")


def take_key(data: dict) -> str | None:
    """Remove every key-ish entry from the inputs and return the first value.

    They must not survive in the dict: inputs are interpolated into task
    descriptions, so an unconsumed key would be pasted into a prompt and sent
    to the model provider.
    """
    found = None
    for name in KEY_INPUTS:
        value = data.pop(name, None)
        if value and not found:
            found = str(value).strip()
    return found


def read_gemini_key() -> str | None:
    for name in KEY_ENV_NAMES:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def set_gemini_key(key: str | None) -> str | None:
    """Publish `key` where the provider will read it. Returns where the effective
    key came from -- "request", "env", or None if there isn't one."""
    if key:
        os.environ["GEMINI_API_KEY"] = key
        # Only touch GOOGLE_API_KEY if something already set it: it wins on
        # precedence, so a stale one would shadow ours. Setting both when it is
        # absent just makes the provider warn on every call.
        if os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = key
        return "request"
    return "env" if read_gemini_key() else None


def fingerprint(key: str | None) -> str:
    """A stable, non-reversible tag for a key, so a log line can say *which* key
    arrived without ever carrying the key itself."""
    if not key:
        return "none"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    return f"sha256:{digest} len={len(key)}"


def credential_report(source: str | None) -> str:
    """One line, safe for the platform's log tab: names and shapes, never values."""
    related = sorted(
        n for n in os.environ
        if any(t in n.upper() for t in ("GEMINI", "GOOGLE", "VERTEX", "GENAI"))
    )
    return (
        f"gemini key source={source or 'NOWHERE'} "
        f"{fingerprint(read_gemini_key())} "
        f"related_env={related or '[]'} total_env={len(os.environ)}"
    )


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

    @before_kickoff
    def resolve_credentials(self, inputs: dict | None) -> dict:
        """Take the key off the request and publish it before any agent runs.

        This lives here, not in main.py, because the platform does not call
        main.kickoff(): it builds this class itself and calls Crew.kickoff()
        directly -- proven by a trace whose Crew Input still carried
        gemini_api_key, a field main.py pops. before_kickoff runs inside
        Crew.kickoff(), so it fires whichever entry point is used.

        The agents were already built, with no key, when the platform
        introspected the crew. That is fine: the Gemini provider defers
        creating its client and re-reads os.environ on the first call, so a
        key published here still lands.
        """
        data = dict(inputs or {})
        source = set_gemini_key(take_key(data))
        print(f"edd-dossier: {credential_report(source)}", flush=True)
        if source is None:
            # Safe to raise: this runs per request, inside kickoff. Raising from
            # _llm() instead would fire at startup and crash-loop the pod.
            raise ValueError(
                "No Gemini API key available. The container has none injected, so "
                "send one with the request: POST /kickoff with \"gemini_api_key\" "
                "inside the \"inputs\" object. " + credential_report(None)
            )
        return data

    def _llm(self) -> LLM:
        # Pass the key explicitly when we have one. Never raise here, even with no
        # key: this runs at startup -- the platform builds the crew to read the
        # {placeholders} for its Inputs tab -- and raising crash-loops the pod and
        # rolls the deploy back. A keyless LLM is fine to construct; the provider
        # defers building its client and re-reads os.environ on the first call, so
        # a key that arrives later with the request still lands.
        key = read_gemini_key()
        kwargs = {"api_key": key} if key else {}
        return LLM(model=self.LLM_MODEL, temperature=0.2, **kwargs)

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
