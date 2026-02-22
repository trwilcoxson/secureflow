# SecureFlow: Product Security Review Agent

A multi-agent product security triage system that evaluates feature descriptions for security, privacy, and GRC (Governance, Risk, Compliance) risks, then creates GitHub issues for the appropriate review teams.

**Project 6 — Agentic AI Systems** | Udacity ND608 Capstone

## Overview

SecureFlow automates the security triage step in feature development. When a developer creates a feature request issue on GitHub and labels it `feature-request`, SecureFlow automatically:

1. Reads the feature description from the GitHub issue
2. Runs three specialist agents in parallel (Security, Privacy, GRC)
3. Creates targeted review issues for teams that need to investigate
4. Posts a summary comment with a GO / CONDITIONAL / NO-GO recommendation

## Architecture

```
GitHub Issue (feature-request label)
        |
        v
  GitHub Action Workflow
        |
        v
    Orchestrator (asyncio.gather)
      /     |     \
     v      v      v
  Security  Privacy  GRC
   Agent    Agent   Agent
  (STRIDE)  (GDPR)  (SOC2)
     \      |      /
      v     v     v
   Review Issues (gh issue create)
   Summary Comment on Source Issue
```

- **Security Agent**: STRIDE threat modeling + OWASP Top 10 mapping
- **Privacy Agent**: Data handling assessment against GDPR, CCPA, HIPAA
- **GRC Agent**: SOC 2, PCI-DSS, ISO 27001 control mapping

## Quick Start

### Local Development

```bash
# Clone and set up
git clone https://github.com/trwilcoxson/secureflow.git
cd secureflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (copy .env.example to .env and add your OpenAI API key)
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run demo + evaluation
python agentic_system.py

# Analyze a specific feature
python agentic_system.py --feature-description "Add user authentication with OAuth2..."

# Run evaluation suite only
python agentic_system.py --evaluate

# Generate PDF report
python generate_report.py
```

### GitHub Action (Production)

The system runs automatically when a `feature-request` label is added to any issue in the repository. Configure `OPENAI_API_KEY` as a repository secret.

## CLI Options

| Flag | Description |
|------|-------------|
| `--issue-number N` | Analyze GitHub issue #N |
| `--feature-description "..."` | Analyze inline text |
| `--evaluate` | Run 7-case evaluation suite |
| `--generate-figures` | Regenerate charts from saved results |
| `--dry-run` | Force dry-run mode (no GitHub API calls) |

## Evaluation

Seven test cases spanning the risk spectrum:

| Case | Expected | Description |
|------|----------|-------------|
| Low-risk internal tool | GO | SSO-protected, anonymized data |
| Critical data exposure | NO-GO | Unauthenticated API with SSN/credit cards |
| Third-party integration | CONDITIONAL | SendGrid with PII in logs |
| ML credit scoring | NO-GO | Automated decisions, bias risk |
| Healthcare portal | NO-GO | PHI to OpenAI, no MFA |
| Vague description | Cautious | Insufficient detail |
| CSS change | GO | No backend/data changes |

## Safeguards

- **DRY_RUN=true** by default (no real GitHub API calls locally)
- **Input validation**: 20-10,000 character limits
- **Output validation**: Pydantic schema enforcement
- **No shell injection**: Uses subprocess with argument lists
- **Secret management**: API keys via env vars / GitHub secrets only
- **Label-gated trigger**: Only `feature-request` labeled issues

## Files

| File | Purpose |
|------|---------|
| `agentic_system.py` | Main implementation (agents, tools, orchestrator, eval) |
| `generate_report.py` | PDF report generation |
| `.github/workflows/security-triage.yml` | GitHub Action workflow |
| `figures/` | Architecture diagram + evaluation charts |
| `module_summary.pdf` | Project report |
| `requirements.txt` | Python dependencies |

## Tech Stack

- **Pydantic AI** — Agent framework with structured output
- **pydantic-evals** — Evaluation framework with LLM judges
- **OpenAI gpt-4o-mini** — LLM backend
- **GitHub Actions** — CI/CD deployment
- **gh CLI** — GitHub issue creation
- **fpdf2** — PDF report generation
- **matplotlib** — Visualization

## Author

Tim Wilcoxson — February 2026
