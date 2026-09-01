#!/usr/bin/env python
"""Entry point for local runs. UiPath calls this through the CrewAI deployment.

Note the platform does NOT call kickoff() below: it builds EddDossierCrew itself
and calls Crew.kickoff() directly. Credential handling therefore lives in the
crew's @before_kickoff hook, which runs on both paths. Keep this file thin.
"""
from edd_dossier.crew import EddDossierCrew

DEFAULTS = {
    "applicant": "ABA Integrated Solutions Limited",
    "entity_type": "Corporate",
    "application": "{}",
    "risk_opinion": "{}",
}


def kickoff(inputs: dict | None = None):
    return EddDossierCrew().crew().kickoff(inputs={**DEFAULTS, **(inputs or {})})


def run():
    """Name the CrewAI scaffold expects."""
    return kickoff()


if __name__ == "__main__":
    print(run())
