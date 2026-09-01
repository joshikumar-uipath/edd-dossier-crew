# Enhanced Due Diligence dossier crew

Four agents that assemble an EDD dossier for a retail bank's account-opening case:
adverse media, ownership and registrations, sanctions/PEP context, and a writer who
turns the three into something an onboarding officer can act on.

Called from a UiPath Maestro case at the Decision stage, after the risk opinion says
enhanced due diligence is required.

**Inputs:** `applicant`, `entity_type`, `application` (JSON string), `risk_opinion` (JSON string)
**Output:** JSON — summary, findings[], residual_risk, recommendation, unresolved[]
