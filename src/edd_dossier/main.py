#!/usr/bin/env python
"""Entry point. UiPath calls this through the CrewAI deployment, passing the
application and the risk opinion the Vertex agent produced in the Decision stage."""
import os

from edd_dossier.crew import EddDossierCrew, credential_report, set_gemini_key

DEFAULTS = {
    "applicant": "ABA Integrated Solutions Limited",
    "entity_type": "Corporate",
    "application": "{}",
    "risk_opinion": "{}",
}

# Accepted spellings for the key when it arrives with the request. The platform
# does not declare it on the Inputs tab, so a caller may reasonably guess at any
# of these; take whichever turns up rather than making them match one exactly.
KEY_INPUTS = ("gemini_api_key", "GEMINI_API_KEY", "google_api_key", "GOOGLE_API_KEY")


def _take_key(data: dict) -> str | None:
    """Remove every key-ish input from the crew inputs and return the first value.

    They must not survive into `data`: the inputs dict is interpolated into task
    descriptions, so an unconsumed key would be pasted into a prompt and sent to
    the model provider.
    """
    found = None
    for name in KEY_INPUTS:
        value = data.pop(name, None)
        if value and not found:
            found = str(value).strip()
    return found


def kickoff(inputs: dict | None = None):
    data = {**DEFAULTS, **(inputs or {})}

    # Neither the Environment Variables page nor an LLM Connection puts a key into
    # this container -- verified from inside it: 27 vars, EXA_API_KEY and
    # INTERNAL_API_KEY among them, neither of ours. So accept it with the request
    # too. Set it before the crew is built; the provider also re-reads os.environ
    # when it first calls out, so a key set here is picked up either way.
    source = set_gemini_key(_take_key(data))
    print(f"edd-dossier: {credential_report(source)}", flush=True)

    if source is None:
        # Safe to raise here -- this is request scoped. Do NOT move this into the
        # crew: _llm() runs at startup, and raising there crash-loops the pod
        # (exit 10) and rolls the deployment back.
        raise ValueError(
            "No Gemini API key available. The container has none injected, so send "
            "one with the request: POST /kickoff with \"gemini_api_key\" inside the "
            "\"inputs\" object. " + credential_report(None)
        )

    return EddDossierCrew().crew().kickoff(inputs=data)


def run():
    """Name the CrewAI scaffold expects."""
    return kickoff()


if __name__ == "__main__":
    print(run())
