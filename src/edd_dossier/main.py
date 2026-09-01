#!/usr/bin/env python
"""Entry point. UiPath calls this through the CrewAI deployment, passing the
application and the risk opinion the Vertex agent produced in the Decision stage."""
import os

from edd_dossier.crew import EddDossierCrew

DEFAULTS = {
    "applicant": "ABA Integrated Solutions Limited",
    "entity_type": "Corporate",
    "application": "{}",
    "risk_opinion": "{}",
}


def kickoff(inputs: dict | None = None):
    data = {**DEFAULTS, **(inputs or {})}
    # Last resort for the key. Neither the Environment Variables page nor an LLM
    # Connection puts GEMINI_API_KEY into this container -- verified from inside
    # it: 27 env vars, EXA_API_KEY and INTERNAL_API_KEY among them, neither of
    # ours. So let the caller send it with the request. Set it before the crew is
    # built, because the agents resolve their LLM at construction.
    key = data.pop("gemini_api_key", None)
    if key:
        os.environ["GEMINI_API_KEY"] = key
    # Names only, never the value -- this lands in the platform's log tab.
    print("edd-dossier: gemini key from",
          "request" if key else ("env" if os.environ.get("GEMINI_API_KEY") else "NOWHERE"),
          flush=True)
    return EddDossierCrew().crew().kickoff(inputs=data)


def run():
    """Name the CrewAI scaffold expects."""
    return kickoff()


if __name__ == "__main__":
    print(run())
