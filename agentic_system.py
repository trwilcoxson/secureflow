"""
SecureFlow — Product Security Review Agent

A multi-agent system that evaluates feature descriptions for security, privacy,
and GRC risks, then creates GitHub issues for the appropriate review teams.

Usage:
    python agentic_system.py                              # Demo + evaluation
    python agentic_system.py --feature-description "..."  # Analyze inline text
    python agentic_system.py --issue-number 42            # Analyze GitHub issue
    python agentic_system.py --evaluate                   # Run evaluation suite
    python agentic_system.py --generate-figures            # Generate charts only
"""

# =============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =============================================================================

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv()

# Configuration — all secrets loaded from environment variables.
# Locally: set in .env file (gitignored).
# In GitHub Actions: set as repository secrets and injected via workflow env.
OPENAI_MODEL = f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
GITHUB_REPO = os.getenv("GITHUB_REPO", "trwilcoxson/secureflow")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [secureflow] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("secureflow")

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SecureFlow Product Security Review")
    parser.add_argument("--issue-number", type=int, help="GitHub issue number to analyze")
    parser.add_argument("--feature-description", type=str, help="Inline feature description")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Override DRY_RUN")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--generate-figures", action="store_true", help="Generate figures only")
    return parser.parse_args()


# =============================================================================
# SECTION 2: PYDANTIC MODELS
# =============================================================================

class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    informational = "informational"


class RiskCategory(str, Enum):
    security = "security"
    privacy = "privacy"
    grc = "grc"


# --- Security ---
class SecurityFinding(BaseModel):
    title: str = Field(description="Short title of the security risk")
    description: str = Field(description="Description of the security concern")
    severity: Severity = Field(description="Severity level of the risk")
    risk_category: str = Field(description="Category of risk (e.g., data exposure, authentication gap, injection surface, third-party dependency)")
    affected_component: str = Field(description="The component or feature area affected")
    recommendation: str = Field(description="Concise recommendation for the security review team")


class SecurityAnalysis(BaseModel):
    summary: str = Field(description="Executive summary of security findings")
    findings: List[SecurityFinding] = Field(description="List of identified security findings")
    overall_risk_level: Severity = Field(description="Overall security risk level")
    requires_review: bool = Field(description="Whether product security team review is warranted")


# --- Privacy ---
class PrivacyFinding(BaseModel):
    title: str = Field(description="Short title of the privacy risk")
    description: str = Field(description="Description of the privacy concern")
    severity: Severity = Field(description="Severity level of the risk")
    data_category: str = Field(description="Category of data affected (e.g., PII, PHI, behavioral, financial)")
    regulation: str = Field(description="Relevant regulation if applicable (e.g., GDPR, CCPA, HIPAA)")
    privacy_principle: str = Field(description="Privacy principle at risk (e.g., data minimization, consent)")
    recommendation: str = Field(description="Concise recommendation for the privacy team")


class PrivacyAnalysis(BaseModel):
    summary: str = Field(description="Executive summary of privacy findings")
    findings: List[PrivacyFinding] = Field(description="List of identified privacy findings")
    overall_risk_level: Severity = Field(description="Overall privacy risk level")
    requires_review: bool = Field(description="Whether privacy team review is warranted")
    data_flow_concerns: str = Field(description="Summary of data flow and handling concerns")


# --- GRC ---
class GRCFinding(BaseModel):
    title: str = Field(description="Short title of the GRC risk")
    description: str = Field(description="Description of the compliance or governance concern")
    severity: Severity = Field(description="Severity level of the risk")
    framework: str = Field(description="Relevant compliance framework (e.g., SOC 2, PCI-DSS, HIPAA)")
    risk_type: str = Field(description="Type of risk (e.g., regulatory, audit, contractual)")
    recommendation: str = Field(description="Concise recommendation for the GRC team")


class GRCAnalysis(BaseModel):
    summary: str = Field(description="Executive summary of GRC findings")
    findings: List[GRCFinding] = Field(description="List of identified GRC findings")
    overall_risk_level: Severity = Field(description="Overall GRC risk level")
    requires_review: bool = Field(description="Whether GRC team review is warranted")
    audit_considerations: str = Field(description="Summary of audit or compliance considerations")


# --- GitHub Issue Result ---
class GitHubIssueResult(BaseModel):
    success: bool = Field(description="Whether the issue was created successfully")
    issue_url: str = Field(default="", description="URL of the created issue")
    issue_number: int = Field(default=0, description="Number of the created issue")
    title: str = Field(description="Title of the created issue")
    labels: List[str] = Field(description="Labels applied to the issue")
    assignee_team: str = Field(description="Team the issue is routed to")
    dry_run: bool = Field(description="Whether this was a dry-run (no real issue created)")
    error: str = Field(default="", description="Error message if creation failed")


# --- Aggregate Review Summary ---
class ReviewSummary(BaseModel):
    feature_name: str = Field(description="Name of the feature being reviewed")
    feature_description: str = Field(description="Description of the feature")
    timestamp: str = Field(description="ISO 8601 timestamp of the review")
    security_analysis: SecurityAnalysis = Field(description="Security analysis results")
    privacy_analysis: PrivacyAnalysis = Field(description="Privacy analysis results")
    grc_analysis: GRCAnalysis = Field(description="GRC analysis results")
    github_issues: List[GitHubIssueResult] = Field(default_factory=list, description="Created GitHub issues")
    total_findings: int = Field(default=0, description="Total number of findings across all domains")
    critical_findings: int = Field(default=0, description="Number of critical-severity findings")
    overall_recommendation: str = Field(default="", description="GO / CONDITIONAL / NO-GO recommendation")
    executive_summary: str = Field(default="", description="Executive summary of the entire review")


# =============================================================================
# SECTION 3: AGENT DEFINITIONS
# =============================================================================

INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"


def load_instructions(domain: str) -> str:
    """Load screening instructions from the instructions/ directory.

    Each team (security, privacy, GRC) maintains their own instruction file.
    This allows teams to update their screening criteria independently
    without modifying the main system code.
    """
    path = INSTRUCTIONS_DIR / f"{domain}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Instructions file not found: {path}. "
            f"Each team must provide an instructions/{domain}.md file."
        )
    return path.read_text()


SECURITY_INSTRUCTIONS = load_instructions("security")
PRIVACY_INSTRUCTIONS = load_instructions("privacy")
GRC_INSTRUCTIONS = load_instructions("grc")

# Agent instances
security_agent = Agent(
    instructions=SECURITY_INSTRUCTIONS,
    output_type=SecurityAnalysis,
)

privacy_agent = Agent(
    instructions=PRIVACY_INSTRUCTIONS,
    output_type=PrivacyAnalysis,
)

grc_agent = Agent(
    instructions=GRC_INSTRUCTIONS,
    output_type=GRCAnalysis,
)


async def analyze_security(feature_desc: str) -> SecurityAnalysis:
    """Run security triage analysis on a feature description."""
    logger.info(f"Security agent: analyzing {len(feature_desc)} chars")
    start = time.time()
    result = await security_agent.run(feature_desc, model=OPENAI_MODEL)
    output = result.output
    duration = time.time() - start
    logger.info(
        f"Security agent: {len(output.findings)} findings, "
        f"risk={output.overall_risk_level.value}, "
        f"review={'yes' if output.requires_review else 'no'}, "
        f"duration={duration:.1f}s"
    )
    return output


async def analyze_privacy(feature_desc: str) -> PrivacyAnalysis:
    """Run privacy triage analysis on a feature description."""
    logger.info(f"Privacy agent: analyzing {len(feature_desc)} chars")
    start = time.time()
    result = await privacy_agent.run(feature_desc, model=OPENAI_MODEL)
    output = result.output
    duration = time.time() - start
    logger.info(
        f"Privacy agent: {len(output.findings)} findings, "
        f"risk={output.overall_risk_level.value}, "
        f"review={'yes' if output.requires_review else 'no'}, "
        f"duration={duration:.1f}s"
    )
    return output


async def analyze_grc(feature_desc: str) -> GRCAnalysis:
    """Run GRC triage analysis on a feature description."""
    logger.info(f"GRC agent: analyzing {len(feature_desc)} chars")
    start = time.time()
    result = await grc_agent.run(feature_desc, model=OPENAI_MODEL)
    output = result.output
    duration = time.time() - start
    logger.info(
        f"GRC agent: {len(output.findings)} findings, "
        f"risk={output.overall_risk_level.value}, "
        f"review={'yes' if output.requires_review else 'no'}, "
        f"duration={duration:.1f}s"
    )
    return output


# =============================================================================
# SECTION 4: TOOL — GITHUB ISSUE CREATOR
# =============================================================================

def format_issue_body(
    category: RiskCategory,
    analysis: SecurityAnalysis | PrivacyAnalysis | GRCAnalysis,
    feature_name: str,
    source_issue: Optional[int] = None,
) -> str:
    """Format a rich Markdown issue body for a triage review issue."""
    lines = []
    lines.append(f"## SecureFlow Automated Triage: {category.value.upper()} Review")
    lines.append("")

    if source_issue:
        lines.append(f"> Triggered by feature request #{source_issue}")
        lines.append("")

    lines.append(f"**Feature:** {feature_name}")
    lines.append(f"**Overall Risk Level:** {analysis.overall_risk_level.value.upper()}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.append("### Summary")
    lines.append(analysis.summary)
    lines.append("")

    if analysis.findings:
        lines.append("### Findings")
        lines.append("")
        lines.append("| # | Severity | Title | Recommendation |")
        lines.append("|---|----------|-------|----------------|")
        for i, f in enumerate(analysis.findings, 1):
            lines.append(f"| {i} | {f.severity.value.upper()} | {f.title} | {f.recommendation} |")
        lines.append("")

    # Domain-specific extras
    if isinstance(analysis, PrivacyAnalysis) and analysis.data_flow_concerns:
        lines.append("### Data Flow Concerns")
        lines.append(analysis.data_flow_concerns)
        lines.append("")
    elif isinstance(analysis, GRCAnalysis) and analysis.audit_considerations:
        lines.append("### Audit Considerations")
        lines.append(analysis.audit_considerations)
        lines.append("")

    lines.append("---")
    lines.append("*This issue was created automatically by SecureFlow. "
                 "Please conduct a full manual review based on these triage findings.*")

    return "\n".join(lines)


async def create_github_issue(
    category: RiskCategory,
    analysis: SecurityAnalysis | PrivacyAnalysis | GRCAnalysis,
    feature_name: str,
    source_issue: Optional[int] = None,
    dry_run: bool = True,
) -> GitHubIssueResult:
    """Create a GitHub issue for a triage review using gh CLI (no shell injection)."""
    severity_label = f"severity:{analysis.overall_risk_level.value}"
    team_label = f"{category.value}-review"
    labels = [team_label, severity_label, "automated-review"]
    title = f"[{category.value.upper()} REVIEW] {feature_name} — {analysis.overall_risk_level.value.upper()} risk"
    body = format_issue_body(category, analysis, feature_name, source_issue)

    if dry_run:
        logger.info(f"[DRY RUN] Would create issue: {title} | labels: {labels}")
        return GitHubIssueResult(
            success=True,
            title=title,
            labels=labels,
            assignee_team=team_label,
            dry_run=True,
        )

    try:
        # Uses asyncio.create_subprocess_exec (not shell=True) to prevent injection
        cmd = [
            "gh", "issue", "create",
            "--repo", GITHUB_REPO,
            "--title", title,
            "--body", body,
            "--label", ",".join(labels),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(f"Failed to create issue: {error_msg}")
            return GitHubIssueResult(
                success=False, title=title, labels=labels,
                assignee_team=team_label, dry_run=False, error=error_msg,
            )

        issue_url = stdout.decode().strip()
        issue_num = int(issue_url.rstrip("/").split("/")[-1])
        logger.info(f"Created issue #{issue_num}: {title}")
        return GitHubIssueResult(
            success=True, issue_url=issue_url, issue_number=issue_num,
            title=title, labels=labels, assignee_team=team_label, dry_run=False,
        )

    except Exception as e:
        logger.error(f"Exception creating issue: {e}")
        return GitHubIssueResult(
            success=False, title=title, labels=labels,
            assignee_team=team_label, dry_run=False, error=str(e),
        )


async def post_summary_comment(
    issue_number: int,
    review: "ReviewSummary",
    dry_run: bool = True,
) -> None:
    """Post a triage summary comment on the original feature request issue."""
    lines = []
    lines.append("## SecureFlow Triage Summary")
    lines.append("")
    lines.append(f"**Recommendation:** {review.overall_recommendation}")
    lines.append(f"**Total Findings:** {review.total_findings} "
                 f"({review.critical_findings} critical)")
    lines.append("")

    for domain, analysis in [
        ("Security", review.security_analysis),
        ("Privacy", review.privacy_analysis),
        ("GRC", review.grc_analysis),
    ]:
        status = "Review requested" if analysis.requires_review else "No review needed"
        lines.append(f"- **{domain}:** {analysis.overall_risk_level.value.upper()} — {status}")

    lines.append("")
    if review.github_issues:
        lines.append("### Created Review Issues")
        for issue in review.github_issues:
            if issue.issue_url:
                lines.append(f"- {issue.title} → {issue.issue_url}")
            else:
                lines.append(f"- {issue.title} *(dry run)*")

    lines.append("")
    lines.append(review.executive_summary)

    body = "\n".join(lines)

    if dry_run:
        logger.info(f"[DRY RUN] Would comment on issue #{issue_number}")
        return

    try:
        # Uses asyncio.create_subprocess_exec (not shell=True) to prevent injection
        cmd = [
            "gh", "issue", "comment", str(issue_number),
            "--repo", GITHUB_REPO,
            "--body", body,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"Failed to comment on issue #{issue_number}: {stderr.decode().strip()}")
        else:
            logger.info(f"Posted summary comment on issue #{issue_number}")
    except Exception as e:
        logger.error(f"Exception commenting on issue: {e}")


# =============================================================================
# SECTION 5: ORCHESTRATOR
# =============================================================================

# Session memory: accumulates reviews across invocations within the same process
review_history: List[ReviewSummary] = []


async def run_security_review(
    feature_description: str,
    feature_name: str = "Unnamed Feature",
    source_issue: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> ReviewSummary:
    """
    Orchestrator: run all three analyst agents in parallel, create issues
    for any domain requiring review, and return an aggregate ReviewSummary.
    """
    if dry_run is None:
        dry_run = DRY_RUN

    # Input validation
    if len(feature_description) < 20:
        raise ValueError("Feature description must be at least 20 characters.")
    if len(feature_description) > 10_000:
        raise ValueError("Feature description must be at most 10,000 characters.")

    logger.info(f"Starting triage for: {feature_name}")

    # Run all 3 agents in parallel with error isolation
    results = await asyncio.gather(
        analyze_security(feature_description),
        analyze_privacy(feature_description),
        analyze_grc(feature_description),
        return_exceptions=True,
    )

    # Handle per-agent failures: if an agent raised an exception, substitute
    # a conservative fallback analysis that flags the domain for manual review.
    def _fallback_security(err: Exception) -> SecurityAnalysis:
        logger.error(f"Security agent failed: {err}")
        return SecurityAnalysis(
            summary=f"Security agent failed ({err}). Manual review required.",
            findings=[], overall_risk_level=Severity.high, requires_review=True,
        )

    def _fallback_privacy(err: Exception) -> PrivacyAnalysis:
        logger.error(f"Privacy agent failed: {err}")
        return PrivacyAnalysis(
            summary=f"Privacy agent failed ({err}). Manual review required.",
            findings=[], overall_risk_level=Severity.high, requires_review=True,
            data_flow_concerns="Unable to assess — agent failure.",
        )

    def _fallback_grc(err: Exception) -> GRCAnalysis:
        logger.error(f"GRC agent failed: {err}")
        return GRCAnalysis(
            summary=f"GRC agent failed ({err}). Manual review required.",
            findings=[], overall_risk_level=Severity.high, requires_review=True,
            audit_considerations="Unable to assess — agent failure.",
        )

    fallbacks = [_fallback_security, _fallback_privacy, _fallback_grc]
    resolved = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            resolved.append(fallbacks[i](r))
        else:
            resolved.append(r)
    security_result, privacy_result, grc_result = resolved

    # Create issues for domains requiring review
    github_issues: List[GitHubIssueResult] = []
    for category, analysis in [
        (RiskCategory.security, security_result),
        (RiskCategory.privacy, privacy_result),
        (RiskCategory.grc, grc_result),
    ]:
        if analysis.requires_review:
            issue = await create_github_issue(
                category, analysis, feature_name, source_issue, dry_run
            )
            github_issues.append(issue)

    # Aggregate findings
    all_findings = (
        security_result.findings + privacy_result.findings + grc_result.findings
    )
    total = len(all_findings)
    critical = sum(1 for f in all_findings if f.severity == Severity.critical)

    # Determine recommendation
    if critical > 0 or any(
        a.overall_risk_level in (Severity.critical, Severity.high)
        for a in (security_result, privacy_result, grc_result)
    ):
        recommendation = "NO-GO"
    elif total > 0:
        recommendation = "CONDITIONAL"
    else:
        recommendation = "GO"

    # Build executive summary
    domains_flagged = []
    if security_result.requires_review:
        domains_flagged.append("security")
    if privacy_result.requires_review:
        domains_flagged.append("privacy")
    if grc_result.requires_review:
        domains_flagged.append("GRC")

    if domains_flagged:
        exec_summary = (
            f"SecureFlow triage identified {total} findings across "
            f"{len(domains_flagged)} domain(s) ({', '.join(domains_flagged)}). "
            f"Recommendation: {recommendation}. "
            f"{'Critical issues require immediate attention. ' if critical > 0 else ''}"
            f"Review issues have been {'created' if not dry_run else 'prepared (dry run)'}."
        )
    else:
        exec_summary = (
            f"SecureFlow triage found no significant risks. "
            f"Recommendation: {recommendation}. No team reviews required."
        )

    review = ReviewSummary(
        feature_name=feature_name,
        feature_description=feature_description,
        timestamp=datetime.now(timezone.utc).isoformat(),
        security_analysis=security_result,
        privacy_analysis=privacy_result,
        grc_analysis=grc_result,
        github_issues=github_issues,
        total_findings=total,
        critical_findings=critical,
        overall_recommendation=recommendation,
        executive_summary=exec_summary,
    )

    # Post summary comment if triggered from a GitHub issue
    if source_issue is not None:
        await post_summary_comment(source_issue, review, dry_run)

    # Session memory
    review_history.append(review)
    logger.info(
        f"Triage complete: {total} findings, {critical} critical, "
        f"recommendation={recommendation}"
    )

    return review


# =============================================================================
# SECTION 6: GITHUB ISSUE READER
# =============================================================================

async def read_github_issue(issue_number: int) -> tuple[str, str]:
    """Read a GitHub issue title and body using gh CLI (no shell injection)."""
    cmd = [
        "gh", "issue", "view", str(issue_number),
        "--repo", GITHUB_REPO,
        "--json", "title,body",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to read issue #{issue_number}: {stderr.decode().strip()}")

    data = json.loads(stdout.decode())
    feature_name = data.get("title", f"Issue #{issue_number}")
    feature_desc = data.get("body", "")

    if not feature_desc:
        raise ValueError(f"Issue #{issue_number} has no body/description.")

    logger.info(f"Read issue #{issue_number}: {feature_name} ({len(feature_desc)} chars)")
    return feature_name, feature_desc


# =============================================================================
# SECTION 7: LOGGING HELPERS (GitHub Actions annotations)
# =============================================================================

def actions_log(level: str, message: str) -> None:
    """Emit GitHub Actions annotations when running in CI."""
    if os.getenv("GITHUB_ACTIONS"):
        if level == "warning":
            print(f"::warning::{message}")
        elif level == "error":
            print(f"::error::{message}")
        else:
            print(f"::notice::{message}")


def print_review_summary(review: ReviewSummary) -> None:
    """Print a human-readable review summary to stdout."""
    print("\n" + "=" * 70)
    print(f"  SECUREFLOW TRIAGE REPORT: {review.feature_name}")
    print("=" * 70)
    print(f"  Timestamp: {review.timestamp}")
    print(f"  Recommendation: {review.overall_recommendation}")
    print(f"  Total Findings: {review.total_findings} ({review.critical_findings} critical)")
    print("-" * 70)

    for domain, analysis in [
        ("SECURITY", review.security_analysis),
        ("PRIVACY", review.privacy_analysis),
        ("GRC", review.grc_analysis),
    ]:
        review_status = "REVIEW REQUESTED" if analysis.requires_review else "No review needed"
        print(f"\n  [{domain}] Risk: {analysis.overall_risk_level.value.upper()} | {review_status}")
        print(f"  Summary: {analysis.summary[:120]}...")
        for i, f in enumerate(analysis.findings, 1):
            print(f"    {i}. [{f.severity.value.upper()}] {f.title}")

    if review.github_issues:
        print(f"\n  GITHUB ISSUES CREATED: {len(review.github_issues)}")
        for issue in review.github_issues:
            mode = "(dry run)" if issue.dry_run else issue.issue_url
            print(f"    - {issue.title} {mode}")

    print("\n" + "-" * 70)
    print(f"  {review.executive_summary}")
    print("=" * 70 + "\n")

    # GitHub Actions annotations
    actions_log("notice", f"SecureFlow: {review.overall_recommendation} — {review.total_findings} findings")
    if review.critical_findings > 0:
        actions_log("warning", f"SecureFlow: {review.critical_findings} CRITICAL findings detected")


# =============================================================================
# SECTION 8: EVALUATION SUITE
# =============================================================================

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge


@dataclass
class HasFindings(Evaluator):
    """Check that the review has at least min_findings total findings."""
    min_findings: int = 1

    async def evaluate(self, ctx: EvaluatorContext[str, ReviewSummary]) -> bool:
        return ctx.output.total_findings >= self.min_findings


@dataclass
class SeverityCheck(Evaluator):
    """Verify that at least one domain flags the expected minimum severity."""
    expected_min_severity: str = "medium"
    domain: str = "any"

    async def evaluate(self, ctx: EvaluatorContext[str, ReviewSummary]) -> bool:
        severity_order = ["informational", "low", "medium", "high", "critical"]
        min_idx = severity_order.index(self.expected_min_severity)

        analyses = []
        if self.domain in ("security", "any"):
            analyses.append(ctx.output.security_analysis)
        if self.domain in ("privacy", "any"):
            analyses.append(ctx.output.privacy_analysis)
        if self.domain in ("grc", "any"):
            analyses.append(ctx.output.grc_analysis)

        for analysis in analyses:
            level_idx = severity_order.index(analysis.overall_risk_level.value)
            if level_idx >= min_idx:
                return True
        return False


@dataclass
class RequiresReviewCheck(Evaluator):
    """Validate that the correct teams are flagged for review."""
    expected_security: bool = False
    expected_privacy: bool = False
    expected_grc: bool = False

    async def evaluate(self, ctx: EvaluatorContext[str, ReviewSummary]) -> bool:
        return (
            ctx.output.security_analysis.requires_review == self.expected_security
            and ctx.output.privacy_analysis.requires_review == self.expected_privacy
            and ctx.output.grc_analysis.requires_review == self.expected_grc
        )


@dataclass
class HasExecutiveSummary(Evaluator):
    """Check that the executive summary is non-empty and substantive."""

    async def evaluate(self, ctx: EvaluatorContext[str, ReviewSummary]) -> bool:
        return len(ctx.output.executive_summary) > 20


@dataclass
class RecommendationCheck(Evaluator):
    """Check that the overall recommendation matches expected value."""
    expected: str = "GO"

    async def evaluate(self, ctx: EvaluatorContext[str, ReviewSummary]) -> bool:
        return ctx.output.overall_recommendation == self.expected


# Task function for the evaluation framework
async def run_triage_for_eval(inputs: List[str]) -> ReviewSummary:
    """Run the full triage pipeline on an input feature description."""
    assert len(inputs) == 1, "Expected exactly one feature description"
    return await run_security_review(inputs[0], "Eval Feature", dry_run=True)


# Evaluation test cases
eval_cases: List[Case[List[str], ReviewSummary, Any]] = [
    Case(
        name="low_risk_internal_tool",
        inputs=[
            "Internal dashboard tool that exports anonymized, aggregated CSV reports "
            "of product usage metrics. Access restricted to employees behind corporate "
            "SSO. No PII is included in exports. Data is read-only from an existing "
            "analytics warehouse. No new data collection."
        ],
        metadata={"expected_risk": "low"},
        evaluators=(
            SeverityCheck(expected_min_severity="low", domain="any"),
            LLMJudge(
                model="openai:gpt-4o-mini",
                rubric="For an internal dashboard behind SSO that exports anonymized, "
                       "aggregated data with no PII — the triage should find minimal or "
                       "no significant risks. A GO or CONDITIONAL recommendation is "
                       "appropriate. Any findings should be informational or low severity.",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="critical_data_exposure",
        inputs=[
            "Public-facing REST API endpoint that returns customer records including "
            "full name, Social Security Number, date of birth, credit card numbers, "
            "and home address. No authentication required. API is accessible at "
            "/api/v1/customers/{id} with sequential integer IDs. Responses are "
            "unencrypted over HTTP. No rate limiting. No audit logging."
        ],
        metadata={"expected_risk": "critical"},
        evaluators=(
            RecommendationCheck(expected="NO-GO"),
            HasFindings(min_findings=3),
            SeverityCheck(expected_min_severity="critical", domain="security"),
            RequiresReviewCheck(expected_security=True, expected_privacy=True, expected_grc=True),
        ),
    ),
    Case(
        name="third_party_integration",
        inputs=[
            "Integration with SendGrid for transactional email. API keys stored in "
            "application config file (not environment variables). Email templates "
            "include customer first name, order details, and shipping address. "
            "Email open/click tracking enabled. Bounce handling logs include "
            "full email addresses. No data processing agreement with SendGrid."
        ],
        metadata={"expected_risk": "medium"},
        evaluators=(
            HasFindings(min_findings=2),
            SeverityCheck(expected_min_severity="medium", domain="any"),
            LLMJudge(
                model="openai:gpt-4o-mini",
                rubric="The review should identify API key storage as a security risk "
                       "and PII in email logs as a privacy concern.",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="ml_credit_scoring",
        inputs=[
            "Machine learning model for automated credit scoring decisions. Model "
            "uses applicant age, zip code, employment history, and social media "
            "activity as features. Decisions are fully automated with no human review. "
            "Model trained on historical lending data from 2010-2020. Results displayed "
            "to applicants with no explanation of factors. Tracking pixel embedded in "
            "decision notification emails."
        ],
        metadata={"expected_risk": "high"},
        evaluators=(
            RecommendationCheck(expected="NO-GO"),
            HasFindings(min_findings=3),
            SeverityCheck(expected_min_severity="high", domain="privacy"),
            LLMJudge(
                model="openai:gpt-4o-mini",
                rubric="The review should identify automated decision-making risks, "
                       "potential bias from historical data, and lack of explainability "
                       "as significant concerns.",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="healthcare_portal",
        inputs=[
            "Patient portal allowing users to view lab results, schedule appointments, "
            "and message their doctor. Medical records include diagnoses, medications, "
            "and treatment plans. Portal sends patient data to OpenAI API for "
            "'smart summary' feature. No MFA required. Session tokens stored in "
            "localStorage. Password reset via SMS only."
        ],
        metadata={"expected_risk": "critical"},
        evaluators=(
            RecommendationCheck(expected="NO-GO"),
            HasFindings(min_findings=4),
            SeverityCheck(expected_min_severity="critical", domain="any"),
            RequiresReviewCheck(expected_security=True, expected_privacy=True, expected_grc=True),
        ),
    ),
    Case(
        name="vague_description",
        inputs=[
            "We want to add a new feature that lets users share their data with "
            "partners. It will involve some API work and a new database table. "
            "Details TBD but we want to start development next sprint."
        ],
        metadata={"expected_risk": "medium"},
        evaluators=(
            HasFindings(min_findings=1),
            LLMJudge(
                model="openai:gpt-4o-mini",
                rubric="Given the vague description mentioning data sharing with partners, "
                       "the review should flag concerns about insufficient detail and "
                       "recommend clarification before proceeding.",
                include_input=True,
            ),
        ),
    ),
    Case(
        name="cosmetic_css_change",
        inputs=[
            "Update the homepage hero banner background color from #3B82F6 to #2563EB "
            "and increase the heading font size from 32px to 36px. CSS-only change in "
            "the marketing landing page. No backend changes, no data changes, no API "
            "modifications, no JavaScript changes."
        ],
        metadata={"expected_risk": "none"},
        evaluators=(
            RecommendationCheck(expected="GO"),
            RequiresReviewCheck(expected_security=False, expected_privacy=False, expected_grc=False),
        ),
    ),
]


eval_dataset = Dataset[List[str], ReviewSummary, Any](
    cases=eval_cases,
    evaluators=[
        HasExecutiveSummary(),
    ],
)


async def run_evaluation() -> dict:
    """Run the full evaluation suite and return results."""
    logger.info("Starting evaluation suite with 7 test cases...")

    report = await eval_dataset.evaluate(run_triage_for_eval)
    report.print(include_input=False, include_output=False, include_durations=True)

    # Extract results for figure generation
    results = {
        "cases": [],
        "total_pass": 0,
        "total_fail": 0,
    }

    for case_result in report.cases:
        case_data = {
            "name": case_result.name,
            "evaluator_results": [],
            "all_passed": True,
        }
        # assertions is dict[str, EvaluationResult[bool]]
        for eval_name, eval_result in case_result.assertions.items():
            passed = eval_result.value is True
            case_data["evaluator_results"].append({
                "evaluator": eval_name,
                "passed": passed,
                "value": eval_result.value,
            })
            if not passed:
                case_data["all_passed"] = False
        # scores is dict[str, EvaluationResult[int|float]]
        for eval_name, eval_result in case_result.scores.items():
            passed = eval_result.value is not None and eval_result.value > 0.5
            case_data["evaluator_results"].append({
                "evaluator": eval_name,
                "passed": passed,
                "value": eval_result.value,
            })
            if not passed:
                case_data["all_passed"] = False

        if case_data["all_passed"]:
            results["total_pass"] += 1
        else:
            results["total_fail"] += 1
        results["cases"].append(case_data)

    logger.info(f"Evaluation complete: {results['total_pass']}/{len(results['cases'])} cases passed")
    return results


# =============================================================================
# SECTION 9: VISUALIZATION & RESULTS EXPORT
# =============================================================================

def generate_architecture_diagram() -> str:
    """Generate architecture diagram using matplotlib patches and arrows."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1.0, 9.5)

    def add_box(x, y, w, h, label, color, fontsize=9, edgecolor="black"):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=edgecolor, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def add_arrow(x1, y1, x2, y2, label="", color="#444"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color,
                            mutation_scale=15),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, ha="left", fontsize=7,
                    color="#555", style="italic")

    # --- Layer labels (left side) ---
    for ly, txt in [(8.5, "TRIGGER"), (6.3, "ORCHESTRATION"),
                    (3.6, "AGENTS"), (1.2, "LLM"), (-0.3, "OUTPUT")]:
        ax.text(-0.3, ly, txt, ha="right", va="center", fontsize=7,
                color="#999", fontweight="bold", rotation=0)

    # --- Row 1: Trigger ---
    add_box(1.5, 8.0, 3.0, 1.0, "GitHub Issue\n(feature-request label)", "#dbeafe", 9)
    add_box(6.0, 8.0, 3.0, 1.0, "GitHub Action\nWorkflow", "#fef3c7", 9)
    add_arrow(4.5, 8.5, 6.0, 8.5, "label event")

    # --- Row 2: Orchestrator ---
    add_box(3.5, 5.8, 3.5, 1.0, "Orchestrator\n(deterministic Python)", "#e0e7ff", 9)
    add_arrow(7.5, 8.0, 5.25, 6.8, "invoke")

    # --- Row 3: Agents (parallel) ---
    add_box(0.0, 3.0, 3.0, 1.2, "Security Agent", "#fee2e2", 9, "#ef4444")
    add_box(3.8, 3.0, 3.0, 1.2, "Privacy Agent", "#dcfce7", 9, "#22c55e")
    add_box(7.5, 3.0, 3.0, 1.2, "GRC Agent", "#fef9c3", 9, "#eab308")

    # Instruction file labels under each agent
    for ax_x, fname in [(1.5, "instructions/\nsecurity.md"),
                         (5.3, "instructions/\nprivacy.md"),
                         (9.0, "instructions/\ngrc.md")]:
        ax.text(ax_x, 2.7, fname, ha="center", va="top", fontsize=6,
                color="#888", style="italic")

    add_arrow(4.2, 5.8, 1.5, 4.2, "parallel")
    add_arrow(5.25, 5.8, 5.3, 4.2, "parallel")
    add_arrow(6.3, 5.8, 9.0, 4.2, "parallel")

    # --- Row 4: LLM ---
    add_box(3.2, 0.8, 4.0, 0.8, "OpenAI API (gpt-4o-mini)", "#f3e8ff", 8)
    add_arrow(1.5, 3.0, 4.5, 1.6)
    add_arrow(5.3, 3.0, 5.3, 1.6)
    add_arrow(9.0, 3.0, 6.5, 1.6)

    # --- Row 5: Outputs (below LLM) ---
    add_box(0.5, -0.8, 3.5, 1.0, "Review Issues\n(gh issue create)", "#dbeafe", 8)
    add_box(5.5, -0.8, 3.5, 1.0, "Summary Comment\non Source Issue", "#f0fdf4", 8)

    # Return arrows: agents return to orchestrator, orchestrator creates outputs
    # Show as: agents -> (return structured output) -> orchestrator -> outputs
    # Use a "results return" label on the parallel arrows (already shown)
    # Then orchestrator -> outputs via side paths
    ax.annotate("", xy=(2.25, 0.2), xytext=(2.25, 0.8),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#444",
                                mutation_scale=15))
    ax.text(2.25, 0.55, "create", ha="center", fontsize=7, color="#555",
            style="italic")
    ax.annotate("", xy=(7.25, 0.2), xytext=(7.25, 0.8),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#444",
                                mutation_scale=15))
    ax.text(7.25, 0.55, "post", ha="center", fontsize=7, color="#555",
            style="italic")

    ax.set_title("SecureFlow Architecture", fontsize=14, fontweight="bold",
                 pad=15, loc="center")

    path = os.path.join(FIGURES_DIR, "fig1_architecture_diagram.png")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated: {path}")
    return path


def generate_eval_results_chart(eval_results: dict) -> str:
    """Generate bar chart of evaluation pass/fail per test case."""
    names = [c["name"].replace("_", "\n") for c in eval_results["cases"]]
    pass_counts = []
    fail_counts = []
    for c in eval_results["cases"]:
        passed = sum(1 for e in c["evaluator_results"] if e["passed"])
        failed = sum(1 for e in c["evaluator_results"] if not e["passed"])
        pass_counts.append(passed)
        fail_counts.append(failed)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, pass_counts, width, label="Pass", color="#22c55e")
    ax.bar(x + width / 2, fail_counts, width, label="Fail", color="#ef4444")
    ax.set_xlabel("Test Case")
    ax.set_ylabel("Evaluator Count")
    ax.set_title("Evaluation Results by Test Case", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.legend()
    ax.set_ylim(0, max(max(pass_counts), max(fail_counts)) + 2)

    path = os.path.join(FIGURES_DIR, "fig2_evaluation_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated: {path}")
    return path


def generate_severity_distribution(review: ReviewSummary) -> str:
    """Generate stacked bar chart of finding severities across domains."""
    domains = ["Security", "Privacy", "GRC"]
    severities = ["critical", "high", "medium", "low", "informational"]
    colors = {"critical": "#dc2626", "high": "#f97316", "medium": "#eab308",
              "low": "#22c55e", "informational": "#60a5fa"}

    data = {s: [] for s in severities}
    analyses = [review.security_analysis, review.privacy_analysis, review.grc_analysis]

    for analysis in analyses:
        counts = {s: 0 for s in severities}
        for f in analysis.findings:
            counts[f.severity.value] += 1
        for s in severities:
            data[s].append(counts[s])

    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = np.zeros(len(domains))
    for sev in severities:
        vals = np.array(data[sev])
        ax.bar(x, vals, 0.5, label=sev.capitalize(), bottom=bottom, color=colors[sev])
        bottom += vals

    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of Findings")
    ax.set_title("Finding Severity Distribution by Domain", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend(loc="upper right")

    path = os.path.join(FIGURES_DIR, "fig3_finding_severity_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated: {path}")
    return path


def generate_sample_output(review: ReviewSummary) -> str:
    """Generate a formatted text rendering of a complete review."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    lines = []
    lines.append(f"SECUREFLOW TRIAGE REPORT")
    lines.append(f"Feature: {review.feature_name}")
    lines.append(f"Recommendation: {review.overall_recommendation}")
    lines.append(f"Total: {review.total_findings} findings ({review.critical_findings} critical)")
    lines.append("")

    for domain, analysis in [
        ("Security", review.security_analysis),
        ("Privacy", review.privacy_analysis),
        ("GRC", review.grc_analysis),
    ]:
        status = "REVIEW" if analysis.requires_review else "OK"
        lines.append(f"[{domain}] {analysis.overall_risk_level.value.upper()} — {status}")
        for f in analysis.findings[:3]:
            lines.append(f"  [{f.severity.value.upper()}] {f.title}")
        if len(analysis.findings) > 3:
            lines.append(f"  ... +{len(analysis.findings) - 3} more")
        lines.append("")

    lines.append(review.executive_summary[:200])

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    path = os.path.join(FIGURES_DIR, "fig4_sample_review_output.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated: {path}")
    return path


def generate_figures(review: ReviewSummary, eval_results: Optional[dict] = None) -> None:
    """Generate all visualization figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    generate_architecture_diagram()
    if eval_results:
        generate_eval_results_chart(eval_results)
    generate_severity_distribution(review)
    generate_sample_output(review)
    logger.info("All figures generated.")


def save_results_json(review: ReviewSummary, eval_results: Optional[dict] = None) -> None:
    """Export ReviewSummary and eval results as JSON."""
    data = {
        "review": review.model_dump(mode="json"),
        "eval_results": eval_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved to {RESULTS_FILE}")


# =============================================================================
# SECTION 10: DEMO FEATURE & MAIN ENTRY POINT
# =============================================================================

DEMO_FEATURE = """
Payment Processing Integration with Stripe

We are building a new payment processing module that integrates with the Stripe API
for handling customer transactions. Key details:

- Accepts credit card payments (Visa, Mastercard, Amex) via Stripe Elements embedded
  in our checkout page.
- Stores a tokenized card reference (Stripe token) in our PostgreSQL database for
  recurring billing.
- Customer billing data (name, email, last 4 digits, billing address) is stored in
  our `billing_profiles` table.
- Webhook endpoint at /api/webhooks/stripe receives payment events (charge.succeeded,
  charge.failed, refund.created) and updates order status.
- Admin dashboard displays transaction history with customer name and partial card number.
- Retry logic for failed charges: up to 3 automatic retries over 7 days.
- Email receipts sent via SendGrid with order details and billing address.
- No encryption at rest for the billing_profiles table currently.
- API keys for Stripe and SendGrid stored in environment variables.
- Logging includes full request/response bodies for debugging (will be turned off
  in production).
"""


async def run_demo() -> ReviewSummary:
    """Run a demo review on the sample payment feature."""
    logger.info("Running demo review on sample payment feature...")
    review = await run_security_review(
        DEMO_FEATURE.strip(),
        "Payment Processing Integration",
        dry_run=True,
    )
    print_review_summary(review)
    return review


async def main():
    args = parse_args()

    # Override dry_run from CLI
    global DRY_RUN
    if args.dry_run is not None:
        DRY_RUN = True

    if args.issue_number:
        # GitHub Action mode: read issue, run triage, create review issues
        feature_name, feature_desc = await read_github_issue(args.issue_number)
        review = await run_security_review(feature_desc, feature_name, source_issue=args.issue_number)
        print_review_summary(review)

    elif args.feature_description:
        # CLI mode: run triage on inline text
        review = await run_security_review(args.feature_description, "CLI Feature")
        print_review_summary(review)

    elif args.evaluate:
        # Evaluation mode
        eval_results = await run_evaluation()
        save_results_json(ReviewSummary(
            feature_name="Evaluation",
            feature_description="N/A",
            timestamp=datetime.now(timezone.utc).isoformat(),
            security_analysis=SecurityAnalysis(summary="N/A", findings=[], overall_risk_level=Severity.informational, requires_review=False),
            privacy_analysis=PrivacyAnalysis(summary="N/A", findings=[], overall_risk_level=Severity.informational, requires_review=False, data_flow_concerns="N/A"),
            grc_analysis=GRCAnalysis(summary="N/A", findings=[], overall_risk_level=Severity.informational, requires_review=False, audit_considerations="N/A"),
        ), eval_results)

    elif args.generate_figures:
        # Generate figures from saved results
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                data = json.load(f)
            review = ReviewSummary(**data["review"])
            generate_figures(review, data.get("eval_results"))
        else:
            logger.error("No results.json found. Run demo first.")
            sys.exit(1)

    else:
        # Default: demo + evaluation + figures
        review = await run_demo()
        eval_results = await run_evaluation()
        generate_figures(review, eval_results)
        save_results_json(review, eval_results)
        logger.info("Demo complete. Figures saved to figures/. Results saved to results.json.")


if __name__ == "__main__":
    asyncio.run(main())
