#!/usr/bin/env python
"""Entry point. UiPath calls this through the CrewAI deployment, passing the
application and the risk opinion the Vertex agent produced in the Decision stage."""
from edd_dossier.crew import EddDossierCrew

DEFAULTS = {
    "applicant": "ABA Integrated Solutions Limited",
    "entity_type": "Corporate",
    "application": "{}",
    "risk_opinion": "{}",
}


def kickoff(inputs: dict | None = None):
    data = {**DEFAULTS, **(inputs or {})}
    return EddDossierCrew().crew().kickoff(inputs=data)


if __name__ == "__main__":
    print(kickoff())
