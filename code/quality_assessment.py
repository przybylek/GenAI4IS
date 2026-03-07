"""
GRADE-CERQual Preparation Script (v2 — Code-Level Analysis)

This script assists with preparing data for GRADE-CERQual assessment by:
1. Extracting citations from LaTeX files for each finding CODE (paragraph level)
2. Aggregating quality assessment ratings
3. Generating Evidence Profile templates

Changes from v1:
- Analysis granularity changed from category level (e.g. C1) to code level
  (e.g. C1.1 "Biases and Discrimination").
- Fixed citation extraction to capture \\parencite*, \\citeauthor, and
  optional biblatex arguments.
- Replaced three hardcoded per-theme parsers with a single generic
  parse_themed_section() that auto-discovers categories (\\subsubsection)
  and codes (\\paragraph).
- Fixed duplicate-counting bug in evidence profile generation by using sets.
- Added LaTeX comment stripping before parsing.
- Improved rating normalisation for case-insensitive matching.
- Cleaned up unused imports.

NOTE: This script CANNOT perform the actual CERQual assessment, which
requires human judgment.  It only prepares the data inputs to support
manual assessment.

Based on GRADE-CERQual guidance:
- Lewin et al. (2015) PLoS Medicine
- Lewin et al. (2018) Implementation Science series

"""

import re
import csv
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class StudyInfo:
    """Information about an included study."""
    study_id: str                                    # e.g. "S01", "R05"
    study_type: str                                  # "secondary" or "agenda"
    # DARE criteria (secondary studies)
    dare_inclusion_criteria: Optional[str] = None
    dare_search_coverage: Optional[str] = None
    dare_quality_assessment: Optional[str] = None
    dare_data_description: Optional[str] = None
    dare_synthesis: Optional[str] = None
    # Custom criteria (research agenda papers)
    agenda_transparency: Optional[str] = None
    agenda_grounding: Optional[str] = None
    agenda_specificity: Optional[str] = None
    agenda_perspectives: Optional[str] = None
    agenda_delimitation: Optional[str] = None


@dataclass
class Finding:
    """A synthesized finding at the *code* level.

    For sections that contain \\paragraph subdivisions inside a
    \\subsubsection, each paragraph becomes a separate Finding with an
    id such as "C1.1".  When a \\subsubsection has no inner paragraphs
    the category itself is treated as a single code (id = "C1").
    """
    code_id: str                  # e.g. "C1.1" or "B1"
    code_name: str                # e.g. "Biases and Discrimination"
    parent_category_id: str       # e.g. "C1"
    parent_category_name: str     # e.g. "Societal, Ethical and Fairness Concerns"
    theme: str                    # "benefit", "challenge", or "future"
    contributing_studies: Set[str] = field(default_factory=set)
    summary_text: str = ""
    # CERQual component assessments (to be filled manually)
    methodological_limitations: str = ""
    coherence: str = ""
    adequacy: str = ""
    relevance: str = ""
    overall_confidence: str = ""
    explanation: str = ""


# ============================================================================
# Constants
# ============================================================================

VALID_RATINGS = {"Yes", "Partly", "No", "Unclear", ""}

QUALITY_SCORE_MAP: Dict[str, float] = {
    "Yes": 1.0,
    "Partly": 0.5,
    "No": 0.0,
    "Unclear": 0.0,
}

TIER_THRESHOLDS: List[Tuple[float, str]] = [
    (0.8, "High"),
    (0.5, "Moderate"),
    (0.2, "Low"),
]                       # anything below 0.2 → "Very Low"

TIER_ABBREVIATIONS: Dict[str, str] = {
    "High": "H",
    "Moderate": "M",
    "Low": "L",
    "Very Low": "VL",
    "Not Rated": "NR",
}

# ============================================================================
# Utility helpers
# ============================================================================

def normalize_rating(rating: str) -> str:
    """Normalize a quality-assessment rating to its canonical form."""
    if rating is None:
        return ""
    rating = rating.strip()
    lower = rating.lower()
    if lower == "yes":
        return "Yes"
    if lower in ("partly", "partial", "partially"):
        return "Partly"
    if lower == "no":
        return "No"
    if lower == "unclear":
        return "Unclear"
    if lower == "":
        return ""
    return rating          # unrecognised – will trigger a warning later


def strip_latex_comments(content: str) -> str:
    """Remove LaTeX comments (% not preceded by \\)."""
    lines = content.split("\n")
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in lines)


def extract_citations_from_text(text: str) -> Set[str]:
    """Return all S## / R## citation keys found in *text*."""
    citations: Set[str] = set()
    # Each pattern allows an optional star and zero-or-more optional [..] args
    commands = [
        "parencite", "cite", "textcite",
        "citep", "citet", "citeauthor", "autocite",
    ]
    for cmd in commands:
        pattern = rf"\\{cmd}\*?(?:\[[^\]]*\])*\{{([^}}]+)\}}"
        for match in re.findall(pattern, text):
            for key in match.split(","):
                key = key.strip()
                if re.match(r"^[SR]\d{2}$", key):
                    citations.add(key)
    return citations


# ============================================================================
# Generic LaTeX section parser (code-level)
# ============================================================================

def parse_themed_section(content: str) -> Dict[str, dict]:
    """Parse a themed LaTeX section and return code-level findings.

    The function handles two structures transparently:

    1. ``\\subsubsection{Name [ID]}`` **with** ``\\paragraph{Code}``
       entries inside → each paragraph becomes a separate code
       (``ID.1``, ``ID.2``, …).
    2. ``\\subsubsection{Name [ID]}`` **without** inner paragraphs →
       the category itself is the code (id = ``ID``).

    If neither ``\\subsubsection`` nor ``\\paragraph`` markers with
    ``[XX]`` tags are found the function falls back to ``\\paragraph``
    entries that carry the ``[XX]`` marker directly (e.g. in overview
    sections).

    Returns
    -------
    dict
        Mapping *code_id* → ``{"name", "citations", "parent_id",
        "parent_name"}``.
    """
    content = strip_latex_comments(content)
    results: Dict[str, dict] = {}

    # --- Try \\subsubsection{...  [XX]} first --------------------------------
    subsec_pattern = r"\\subsubsection\{(.*?)\s*\[([A-Z]\d+)\]\}"
    subsec_matches = list(re.finditer(subsec_pattern, content))

    if not subsec_matches:
        # Fallback: \\paragraph{... [XX]} (alternative structure)
        para_cat_pattern = r"\\paragraph\{(.*?)\s*\[([A-Z]\d+)\]\}"
        para_cat_matches = list(re.finditer(para_cat_pattern, content))

        for i, m in enumerate(para_cat_matches):
            cat_name = m.group(1).strip()
            cat_id = m.group(2).strip()
            start = m.end()
            if i + 1 < len(para_cat_matches):
                end = para_cat_matches[i + 1].start()
            else:
                nxt = re.search(
                    r"\\(?:section|subsection|subsubsection)\{", content[start:]
                )
                end = start + nxt.start() if nxt else len(content)
            citations = extract_citations_from_text(content[start:end])
            results[cat_id] = {
                "name": cat_name,
                "citations": citations,
                "parent_id": cat_id,
                "parent_name": cat_name,
            }
        return results

    # --- Process each \\subsubsection -----------------------------------------
    for i, m in enumerate(subsec_matches):
        cat_name = m.group(1).strip()
        cat_id = m.group(2).strip()

        sec_start = m.end()
        if i + 1 < len(subsec_matches):
            sec_end = subsec_matches[i + 1].start()
        else:
            nxt = re.search(
                r"\\(?:section|subsection|subsubsection)\{", content[sec_start:]
            )
            sec_end = sec_start + nxt.start() if nxt else len(content)

        section_content = content[sec_start:sec_end]

        # Look for \\paragraph entries inside this subsubsection
        para_pattern = r"\\paragraph\{([^}]+)\}"
        para_matches = list(re.finditer(para_pattern, section_content))

        if para_matches:
            # ---- code-level (paragraph) parsing ----------------------------
            # Any citations *before* the first paragraph are added to code .1
            pre_para_text = section_content[: para_matches[0].start()]
            pre_citations = extract_citations_from_text(pre_para_text)

            for j, pm in enumerate(para_matches):
                code_name = pm.group(1).strip()
                code_id = f"{cat_id}.{j + 1}"

                p_start = pm.end()
                p_end = (
                    para_matches[j + 1].start()
                    if j + 1 < len(para_matches)
                    else len(section_content)
                )
                citations = extract_citations_from_text(section_content[p_start:p_end])

                if j == 0 and pre_citations:
                    citations.update(pre_citations)

                results[code_id] = {
                    "name": code_name,
                    "citations": citations,
                    "parent_id": cat_id,
                    "parent_name": cat_name,
                }
        else:
            # ---- category-level fallback (no inner paragraphs) -------------
            citations = extract_citations_from_text(section_content)
            results[cat_id] = {
                "name": cat_name,
                "citations": citations,
                "parent_id": cat_id,
                "parent_name": cat_name,
            }

    return results


# ============================================================================
# Quality-assessment loading and aggregation
# ============================================================================

def load_quality_assessments(csv_file: Path) -> Dict[str, StudyInfo]:
    """Load quality assessment data from a CSV file.

    Expected columns::

        study_id, study_type,
        dare1, dare2, dare3, dare4, dare5,
        agenda1, agenda2, agenda3, agenda4, agenda5
    """
    studies: Dict[str, StudyInfo] = {}
    if not csv_file.exists():
        print(f"Warning: Quality assessment file not found: {csv_file}")
        return studies

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"  CSV columns found: {reader.fieldnames}")
        for row in reader:
            study = StudyInfo(
                study_id=row["study_id"],
                study_type=row["study_type"],
                dare_inclusion_criteria=normalize_rating(row.get("dare1", "")),
                dare_search_coverage=normalize_rating(row.get("dare2", "")),
                dare_quality_assessment=normalize_rating(row.get("dare3", "")),
                dare_data_description=normalize_rating(row.get("dare4", "")),
                dare_synthesis=normalize_rating(row.get("dare5", "")),
                agenda_transparency=normalize_rating(row.get("agenda1", "")),
                agenda_grounding=normalize_rating(row.get("agenda2", "")),
                agenda_specificity=normalize_rating(row.get("agenda3", "")),
                agenda_perspectives=normalize_rating(row.get("agenda4", "")),
                agenda_delimitation=normalize_rating(row.get("agenda5", "")),
            )
            studies[study.study_id] = study
    return studies


def aggregate_quality_for_finding(
    contributing_studies: Set[str],
    quality_data: Dict[str, StudyInfo],
) -> Dict:
    """Aggregate quality ratings for the studies that contribute to a finding.

    Returns summary statistics that can *inform* (but not determine) the
    methodological-limitations component of CERQual.
    """
    secondary_studies: List[StudyInfo] = []
    agenda_studies: List[StudyInfo] = []

    for sid in contributing_studies:
        if sid in quality_data:
            study = quality_data[sid]
            if study.study_type == "secondary":
                secondary_studies.append(study)
            else:
                agenda_studies.append(study)

    # ---- DARE summary (secondary studies) ----------------------------------
    dare_summary: Dict = {
        "total_secondary": len(secondary_studies),
        "criteria": {
            "inclusion_criteria": {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "search_coverage":    {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "quality_assessment": {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "data_description":   {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "synthesis":          {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
        },
    }
    for study in secondary_studies:
        for criterion, attr in [
            ("inclusion_criteria", "dare_inclusion_criteria"),
            ("search_coverage",    "dare_search_coverage"),
            ("quality_assessment", "dare_quality_assessment"),
            ("data_description",   "dare_data_description"),
            ("synthesis",          "dare_synthesis"),
        ]:
            rating = getattr(study, attr, "") or ""
            if rating in dare_summary["criteria"][criterion]:
                dare_summary["criteria"][criterion][rating] += 1
            elif rating:
                print(f"  Warning: Unexpected DARE rating '{rating}' "
                      f"for {study.study_id}.{criterion}")

    # ---- Agenda summary ----------------------------------------------------
    agenda_summary: Dict = {
        "total_agenda": len(agenda_studies),
        "criteria": {
            "transparency":  {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "grounding":     {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "specificity":   {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "perspectives":  {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
            "delimitation":  {"Yes": 0, "Partly": 0, "No": 0, "Unclear": 0},
        },
    }
    for study in agenda_studies:
        for criterion, attr in [
            ("transparency",  "agenda_transparency"),
            ("grounding",     "agenda_grounding"),
            ("specificity",   "agenda_specificity"),
            ("perspectives",  "agenda_perspectives"),
            ("delimitation",  "agenda_delimitation"),
        ]:
            rating = getattr(study, attr, "") or ""
            if rating in agenda_summary["criteria"][criterion]:
                agenda_summary["criteria"][criterion][rating] += 1
            elif rating:
                print(f"  Warning: Unexpected agenda rating '{rating}' "
                      f"for {study.study_id}.{criterion}")

    return {
        "secondary": dare_summary,
        "agenda": agenda_summary,
        "total_studies": len(contributing_studies),
        "study_ids": sorted(contributing_studies),
    }

def compute_study_quality_score(study: StudyInfo) -> Tuple[float, str]:
    """Return a normalised quality score (0–1) and a quality tier.

    Scoring
    -------
    Each criterion is mapped via ``QUALITY_SCORE_MAP`` (Yes → 1, Partly → 0.5,
    No/Unclear → 0).  The score is the mean over rated criteria.

    Tiers
    -----
    High (≥ 0.8) · Moderate (≥ 0.5) · Low (≥ 0.2) · Very Low (< 0.2).
    """
    if study.study_type == "secondary":
        ratings = [
            study.dare_inclusion_criteria,
            study.dare_search_coverage,
            study.dare_quality_assessment,
            study.dare_data_description,
            study.dare_synthesis,
        ]
    else:
        ratings = [
            study.agenda_transparency,
            study.agenda_grounding,
            study.agenda_specificity,
            study.agenda_perspectives,
            study.agenda_delimitation,
        ]

    valid_scores = [
        QUALITY_SCORE_MAP[r] for r in ratings if r in QUALITY_SCORE_MAP
    ]

    if not valid_scores:
        return 0.0, "Not Rated"

    score = sum(valid_scores) / len(valid_scores)

    for threshold, tier in TIER_THRESHOLDS:
        if score >= threshold:
            return score, tier
    return score, "Very Low"


def assess_auto_confidence(
    mean_score: float,
    n_studies: int,
    tier_counts: Dict[str, int],
) -> Tuple[str, bool, List[str]]:
    """Compute a preliminary CERQual confidence rating and flag for review.

    Auto-confidence is driven by the count of High-quality contributing
    studies.  A finding anchored by several High-quality studies receives
    high confidence regardless of whether a few weaker studies also
    contribute.

    Only two genuinely problematic conditions trigger a review flag:
    thin evidence (too few studies to judge adequacy) and a
    predominantly weak evidence base (majority Low / Very Low).

    Parameters
    ----------
    mean_score : float
        Mean quality score (0-1) across contributing studies.  Retained
        in the output for reference but not used for tier assignment.
    n_studies : int
        Total number of contributing studies.
    tier_counts : dict
        Counts per quality tier (High / Moderate / Low / Very Low).

    Returns
    -------
    auto_confidence : str
        One of "High", "Moderate", "Low", "Very Low".
    needs_review : bool
        True if the code should be manually reviewed.
    reasons : list[str]
        Human-readable reasons for flagging (empty when not flagged).
    """
    n_high = tier_counts.get("High", 0)
    n_moderate = tier_counts.get("Moderate", 0)
    n_low = tier_counts.get("Low", 0)
    n_vl = tier_counts.get("Very Low", 0)
    n_strong = n_high + n_moderate
    n_weak = n_low + n_vl

    reasons: List[str] = []

    # ---- auto-confidence (anchor-driven) ----------------------------------
    if n_high >= 3:
        auto = "High"
    elif (n_high >= 1 and n_strong >= 3) or n_moderate >= 4:
        auto = "Moderate"
    elif n_studies >= 2 and n_strong >= 1:
        auto = "Low"
    else:
        auto = "Very Low"

    # ---- review flags (only genuinely problematic cases) ------------------
    if n_studies <= 2:
        reasons.append(f"Thin evidence (n={n_studies})")

    if n_studies > 2 and (n_weak / n_studies) > 0.50:
        reasons.append(
            f"Predominantly weak ({n_weak}/{n_studies} Low/VL)"
        )

    needs_review = len(reasons) > 0
    return auto, needs_review, reasons

# ============================================================================
# Output generation
# ============================================================================
def generate_evidence_profile_template(
    findings: Dict[str, Finding],
    quality_data: Dict[str, StudyInfo],
    output_file: Path,
) -> None:
    """Generate a CSV template for the GRADE-CERQual Evidence Profile."""

    headers = [
        "Category",
        "Code Name",
        "Contributing Studies (Total)",
        "Secondary Studies",
        "Research Agendas",
        "Study IDs",
        "Mean Quality Score",
        "Quality: High (n)",
        "Quality: Moderate (n)",
        "Quality: Low (n)",
        "Quality: Very Low (n)",
        "Per-Study Quality",
        "Auto Confidence",
        "Needs Review",
        "Review Reasons",
        "Methodological Limitations",
        "ML Explanation",
        "Coherence",
        "Coherence Explanation",
        "Adequacy of Data",
        "Adequacy Explanation",
        "Relevance",
        "Relevance Explanation",
        "Overall Confidence",
        "Overall Explanation",
    ]

    rows: List[list] = []
    for code_id, finding in sorted(findings.items()):
        contributing = sorted(finding.contributing_studies)
        sec_ids = [s for s in contributing if s.startswith("S")]
        ag_ids = [s for s in contributing if s.startswith("R")]

        # ---- per-study quality --------------------------------------------
        study_scores: List[float] = []
        tier_counts: Dict[str, int] = {
            "High": 0, "Moderate": 0, "Low": 0, "Very Low": 0,
        }
        per_study_parts: List[str] = []

        for study_id in contributing:
            if study_id in quality_data:
                score, tier = compute_study_quality_score(
                    quality_data[study_id]
                )
                study_scores.append(score)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                per_study_parts.append(f"{study_id}({score:.1f})")
            else:
                per_study_parts.append(f"{study_id}(N/A)")

        mean_score = (
            sum(study_scores) / len(study_scores) if study_scores else 0.0
        )

        # ---- auto-confidence and review flag ------------------------------
        auto_conf, needs_review, review_reasons = assess_auto_confidence(
            mean_score, len(contributing), tier_counts
        )

        row = [
            finding.parent_category_id,
            finding.code_name,
            len(contributing),
            len(sec_ids),
            len(ag_ids),
            ", ".join(contributing),
            f"{mean_score:.2f}",
            tier_counts["High"],
            tier_counts["Moderate"],
            tier_counts["Low"],
            tier_counts["Very Low"],
            "; ".join(per_study_parts),
            auto_conf,
            "Yes" if needs_review else "No",
            "; ".join(review_reasons) if review_reasons else "",
            "",  # Methodological Limitations
            "",  # ML Explanation
            "",  # Coherence
            "",  # Coherence Explanation
            "",  # Adequacy
            "",  # Adequacy Explanation
            "",  # Relevance
            "",  # Relevance Explanation
            "",  # Overall Confidence
            "",  # Overall Explanation
        ]
        rows.append(row)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # ---- print triage summary ---------------------------------------------
    total = len(rows)
    flagged = sum(1 for r in rows if r[13] == "Yes")
    print(f"Evidence Profile template saved to: {output_file}")
    print(f"  Total codes: {total}")
    print(f"  Flagged for review: {flagged} ({100 * flagged / total:.0f}%)")
    print(f"  Auto-assessed (no flag): {total - flagged} "
          f"({100 * (total - flagged) / total:.0f}%)")



def generate_soqf_template(
    findings: Dict[str, Finding],
    output_file: Path,
) -> None:
    """Generate a CSV template for the Summary of Qualitative Findings."""

    headers = [
        "Summary of Review Finding",
        "CERQual Assessment of Confidence",
        "Explanation of CERQual Assessment",
        "Studies Contributing to the Review Finding",
    ]

    rows: List[list] = []
    for code_id, finding in sorted(findings.items()):
        label = f"[{code_id}] {finding.code_name}"
        rows.append([
            f"{label}: {finding.summary_text or '[TO BE WRITTEN]'}",
            "",
            "",
            ", ".join(sorted(finding.contributing_studies)),
        ])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"SoQF template saved to: {output_file}")


def generate_quality_assessment_template(
    output_file: Path,
    study_ids: List[str],
) -> None:
    """Generate a blank CSV template for entering quality-assessment ratings."""

    headers = [
        "study_id", "study_type",
        "dare1", "dare2", "dare3", "dare4", "dare5",
        "agenda1", "agenda2", "agenda3", "agenda4", "agenda5",
    ]

    rows = []
    for study_id in sorted(study_ids):
        study_type = "secondary" if study_id.startswith("S") else "agenda"
        rows.append([study_id, study_type] + [""] * 10)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Quality assessment template saved to: {output_file}")
    print("\nDARE Criteria (rate as: Yes / Partly / No / Unclear):")
    print("  dare1: Are inclusion/exclusion criteria described and appropriate?")
    print("  dare2: Is the literature search likely to have covered all relevant studies?")
    print("  dare3: Did the reviewers assess quality/validity of included studies?")
    print("  dare4: Were the basic data/studies adequately described?")
    print("  dare5: Were the findings of the relevant studies appropriately synthesized?")
    print("\nCustom Agenda Criteria (rate as: Yes / Partly / No / Unclear):")
    print("  agenda1: Is the method for identifying research gaps transparently described?")
    print("  agenda2: Are proposed directions grounded in cited empirical evidence?")
    print("  agenda3: Are proposed research directions specific and actionable?")
    print("  agenda4: Does the paper consider multiple stakeholder perspectives?")
    print("  agenda5: Is the scope and context of applicability clearly delimited?")


# ============================================================================
# Quality-assessment statistics and reporting
# ============================================================================

def compute_quality_statistics(
    quality_data: Dict[str, StudyInfo],
) -> Dict:
    """Compute summary statistics for quality assessments."""

    stats: Dict = {
        "secondary": {
            "total": 0,
            "dare_criteria": {
                "dare1 (Inclusion Criteria)": defaultdict(int),
                "dare2 (Search Coverage)":    defaultdict(int),
                "dare3 (Quality Assessment)": defaultdict(int),
                "dare4 (Data Description)":   defaultdict(int),
                "dare5 (Synthesis)":          defaultdict(int),
            },
        },
        "agenda": {
            "total": 0,
            "agenda_criteria": {
                "agenda1 (Transparency)":  defaultdict(int),
                "agenda2 (Grounding)":     defaultdict(int),
                "agenda3 (Specificity)":   defaultdict(int),
                "agenda4 (Perspectives)":  defaultdict(int),
                "agenda5 (Delimitation)":  defaultdict(int),
            },
        },
    }

    for study in quality_data.values():
        if study.study_type == "secondary":
            stats["secondary"]["total"] += 1
            for criterion, attr in [
                ("dare1 (Inclusion Criteria)", "dare_inclusion_criteria"),
                ("dare2 (Search Coverage)",    "dare_search_coverage"),
                ("dare3 (Quality Assessment)", "dare_quality_assessment"),
                ("dare4 (Data Description)",   "dare_data_description"),
                ("dare5 (Synthesis)",          "dare_synthesis"),
            ]:
                rating = getattr(study, attr, "") or "Not Rated"
                stats["secondary"]["dare_criteria"][criterion][rating] += 1
        else:
            stats["agenda"]["total"] += 1
            for criterion, attr in [
                ("agenda1 (Transparency)",  "agenda_transparency"),
                ("agenda2 (Grounding)",     "agenda_grounding"),
                ("agenda3 (Specificity)",   "agenda_specificity"),
                ("agenda4 (Perspectives)",  "agenda_perspectives"),
                ("agenda5 (Delimitation)",  "agenda_delimitation"),
            ]:
                rating = getattr(study, attr, "") or "Not Rated"
                stats["agenda"]["agenda_criteria"][criterion][rating] += 1

    return stats


def print_quality_summary(quality_data: Dict[str, StudyInfo]) -> None:
    """Print a formatted summary of quality assessment results."""
    stats = compute_quality_statistics(quality_data)

    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT SUMMARY")
    print("=" * 70)

    print(f"\nSecondary Studies (n={stats['secondary']['total']}):")
    print("-" * 60)
    print(f"{'Criterion':<30} {'Yes':>6} {'Partly':>8} {'No':>6} {'Unclear':>8}")
    print("-" * 60)
    for criterion, counts in stats["secondary"]["dare_criteria"].items():
        print(
            f"{criterion:<30} {counts.get('Yes', 0):>6} "
            f"{counts.get('Partly', 0):>8} {counts.get('No', 0):>6} "
            f"{counts.get('Unclear', 0):>8}"
        )

    print(f"\nResearch Agenda Papers (n={stats['agenda']['total']}):")
    print("-" * 60)
    print(f"{'Criterion':<30} {'Yes':>6} {'Partly':>8} {'No':>6} {'Unclear':>8}")
    print("-" * 60)
    for criterion, counts in stats["agenda"]["agenda_criteria"].items():
        print(
            f"{criterion:<30} {counts.get('Yes', 0):>6} "
            f"{counts.get('Partly', 0):>8} {counts.get('No', 0):>6} "
            f"{counts.get('Unclear', 0):>8}"
        )


# ============================================================================
# LaTeX table generation
# ============================================================================
def escape_latex(text: str) -> str:
    """Escape common LaTeX special characters in plain text."""
    for char in ('&', '%', '$', '#', '_'):
        text = text.replace(char, f'\\{char}')
    return text


def generate_latex_evidence_tables(
    findings: Dict[str, Finding],
    quality_data: Dict[str, StudyInfo],
    output_dir: Path,
) -> None:
    """Generate three LaTeX tables (one per theme) for the evidence profile.

    Adheres to LaTeX good practices and Springer Nature (sn-jnl) template:
    1.  **Row Spacing**: Uses `\\renewcommand{\\arraystretch}{1.25}` to add "slight space" between rows cleanly, without manual hacks.
    2.  **Clean Code**: Removes `\\par` and `\\raggedright` from individual cells to prevent formatting clutter and vertical alignment issues.
    3.  **Scope-based Styling**: Applies `\\footnotesize` to the entire table environment.
    4.  **Template Compliance**: Uses `\\botrule` and `\\midrule`.
    """

    theme_configs = [
        ("benefit",
         "table_certainty_benefits.tex",
         "Certainty of Evidence: Benefits",
         "tab:certainty_benefits"),
        ("challenge",
         "table_certainty_challenges.tex",
         "Certainty of Evidence: Challenges and Limitations",
         "tab:certainty_challenges"),
        ("future",
         "table_certainty_future.tex",
         "Certainty of Evidence: Research Gaps and Future Directions",
         "tab:certainty_future"),
    ]

    for theme, filename, caption, label in theme_configs:
        theme_findings = {
            k: v for k, v in findings.items() if v.theme == theme
        }
        if not theme_findings:
            continue

        # Group by parent category, preserving sort order
        categories: Dict[str, List[Tuple[str, Finding]]] = defaultdict(list)
        for code_id in sorted(theme_findings):
            finding = theme_findings[code_id]
            categories[finding.parent_category_id].append(
                (code_id, finding)
            )

        cat_ids = sorted(categories.keys())

        output_file = output_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"% {caption}\n")
            f.write("\\begin{table}[h]\n")
            f.write(f"\\caption{{{caption}}}\\label{{{label}}}\n")

            # 1. Increase row height (1.25 is standard for "slight space")
            f.write("\\renewcommand{\\arraystretch}{1.25}\n")

            # 2. Apply font size globally
            f.write("\\footnotesize\n")

            # 3. Use tabular* to fill textwidth
            # Columns:
            # l (Category, minimal width)
            # p (Code Name, ~35%)
            # p (Studies, ~55%)
            f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l p{0.35\\textwidth} p{0.55\\textwidth}@{}}\n")

            f.write("\\toprule\n")
            f.write("\\textbf{Cat.} & \\textbf{Code Name} & "
                    "\\textbf{Contributing Studies (Quality Score)}\\\\\n")
            f.write("\\midrule\n")

            for ci, cat_id in enumerate(cat_ids):
                codes = categories[cat_id]

                for j, (code_id, finding) in enumerate(codes):
                    # Construct studies string
                    parts: List[str] = []
                    for sid in sorted(finding.contributing_studies):
                        if sid in quality_data:
                            score, _ = compute_study_quality_score(
                                quality_data[sid]
                            )
                            parts.append(f"{sid}({score:.1f})")
                        else:
                            parts.append(f"{sid}(N/A)")
                    studies_str = "; ".join(parts)

                    # Only display Category ID on the first row of the group
                    cat_cell = cat_id if j == 0 else ""
                    code_name = escape_latex(finding.code_name)

                    # Clean data row (no \par, no inline font commands)
                    f.write(
                        f"{cat_cell} & {code_name} & {studies_str} \\\\\n"
                    )

                # Add separator between categories (but not after the last one)
                if ci < len(cat_ids) - 1:
                    f.write("\\midrule\n")

            # SN Template specific bottom rule
            f.write("\\botrule\n")
            f.write("\\end{tabular*}\n")
            f.write("\\end{table}\n")

        print(f"LaTeX table saved to: {output_file}")


def generate_latex_quality_tables(
    quality_data: Dict[str, StudyInfo],
    output_dir: Path,
) -> None:
    """Generate LaTeX tables for quality assessment results."""

    # ---- Secondary studies (DARE) ------------------------------------------
    secondary_table = output_dir / "table_quality_secondary.tex"
    with open(secondary_table, "w", encoding="utf-8") as f:
        f.write("% DARE Quality Assessment - Secondary Studies\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\caption{Quality Assessment of Secondary Studies "
                "Using DARE Criteria}\\label{tab:quality_secondary}%\n")
        f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Study} & "
                "\\textbf{C1} & "
                "\\textbf{C2} & "
                "\\textbf{C3} & "
                "\\textbf{C4} & "
                "\\textbf{C5} & "
                "\\textbf{Score}\\\\\n")
        f.write("\\midrule\n")

        for sid in sorted(s for s in quality_data if s.startswith("S")):
            s = quality_data[sid]
            score, _ = compute_study_quality_score(s)
            cells = [
                sid,
                s.dare_inclusion_criteria or "--",
                s.dare_search_coverage or "--",
                s.dare_quality_assessment or "--",
                s.dare_data_description or "--",
                s.dare_synthesis or "--",
                f"{score:.1f}",
            ]
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to: {secondary_table}")

    # ---- Research agenda papers --------------------------------------------
    agenda_table = output_dir / "table_quality_agendas.tex"
    with open(agenda_table, "w", encoding="utf-8") as f:
        f.write("% Quality Assessment - Research Agenda Papers\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\caption{Quality Assessment of Research Agenda "
                "Papers}\\label{tab:quality_agendas}%\n")
        f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Study} & "
                "\\textbf{C1} & "
                "\\textbf{C2} & "
                "\\textbf{C3} & "
                "\\textbf{C4} & "
                "\\textbf{C5} & "
                "\\textbf{Score}\\\\\n")
        f.write("\\midrule\n")

        for sid in sorted(s for s in quality_data if s.startswith("R")):
            s = quality_data[sid]
            score, _ = compute_study_quality_score(s)
            cells = [
                sid,
                s.agenda_transparency or "--",
                s.agenda_grounding or "--",
                s.agenda_specificity or "--",
                s.agenda_perspectives or "--",
                s.agenda_delimitation or "--",
                f"{score:.1f}",
            ]
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to: {agenda_table}")






# ============================================================================
# Summary report
# ============================================================================

def generate_summary_report(
    findings: Dict[str, Finding],
    all_studies: Set[str],
    output_file: Path,
) -> None:
    """Write a plain-text summary report."""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("GRADE-CERQual Preparation Summary\n")
        f.write("=" * 50 + "\n\n")

        sec_n = sum(1 for s in all_studies if s.startswith("S"))
        ag_n = sum(1 for s in all_studies if s.startswith("R"))
        f.write(f"Total unique studies: {len(all_studies)}\n")
        f.write(f"  Secondary studies (S##): {sec_n}\n")
        f.write(f"  Research agendas  (R##): {ag_n}\n")

        if findings:
            f.write("\n\nFindings by code:\n")
            f.write("-" * 50 + "\n")

            current_parent: Optional[str] = None
            for code_id in sorted(findings):
                finding = findings[code_id]
                if finding.parent_category_id != current_parent:
                    current_parent = finding.parent_category_id
                    f.write(
                        f"\n[{finding.parent_category_id}] "
                        f"{finding.parent_category_name}\n"
                    )
                f.write(f"\n  {code_id} - {finding.code_name}\n")
                f.write(f"    Theme: {finding.theme}\n")
                n = len(finding.contributing_studies)
                ids = ", ".join(sorted(finding.contributing_studies))
                f.write(f"    Contributing studies ({n}): {ids}\n")
                s_n = sum(1 for s in finding.contributing_studies
                          if s.startswith("S"))
                a_n = sum(1 for s in finding.contributing_studies
                          if s.startswith("R"))
                f.write(f"    - Secondary studies: {s_n}\n")
                f.write(f"    - Research agendas:  {a_n}\n")

        f.write("\n\n" + "=" * 50 + "\n")
        f.write("IMPORTANT REMINDER\n")
        f.write("=" * 50 + "\n")
        f.write(
            """
The CERQual assessment CANNOT be automated. Human judgment is required for:

1. METHODOLOGICAL LIMITATIONS
   - Consider which limitations MATTER for THIS SPECIFIC finding
   - Same limitation may be serious for one finding, minor for another

2. COHERENCE
   - Assess fit between data and the synthesized finding
   - Look for contradictory data, ambiguity, competing explanations

3. ADEQUACY OF DATA
   - Judge BOTH richness AND quantity of data
   - Consider whether additional data would change the finding

4. RELEVANCE
   - Assess applicability to YOUR review context
   - Consider population, phenomenon, setting

5. OVERALL CONFIDENCE
   - Integrate all four components
   - Provide transparent explanation for the judgment

Reference: Lewin et al. (2018) Implementation Science series
"""
        )

    print(f"Summary report saved to: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    project_root = Path(".")
    theme_files = [
        (project_root / "Benefits.tex",                    "benefit"),
        (project_root / "Challenges_and_Limitations.tex",  "challenge"),
        (project_root / "Future_Research.tex",             "future"),
    ]
    quality_file = project_root / "quality_assessments.csv"
    output_dir = project_root / "cerqual_output"

    print("=" * 70)
    print("GRADE-CERQual Preparation Script (v2 — Code-Level Analysis)")
    print("=" * 70)
    print("\nNOTE: This script prepares DATA for CERQual assessment.")
    print("The actual CERQual assessment requires HUMAN JUDGMENT.")
    print("See: Lewin et al. (2018) Implementation Science series.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Parse each themed LaTeX file at the code (paragraph) level
    # ------------------------------------------------------------------
    all_studies: Set[str] = set()
    findings: Dict[str, Finding] = {}

    for tex_file, theme in theme_files:
        if not tex_file.exists():
            print(f"\nWarning: {theme.capitalize()} file not found: {tex_file}")
            continue

        print(f"\nParsing {theme} section: {tex_file}")
        content = tex_file.read_text(encoding="utf-8")
        parsed = parse_themed_section(content)

        for code_id, info in sorted(parsed.items()):
            finding = Finding(
                code_id=code_id,
                code_name=info["name"],
                parent_category_id=info["parent_id"],
                parent_category_name=info["parent_name"],
                theme=theme,
                contributing_studies=info["citations"],
            )
            findings[code_id] = finding
            all_studies.update(info["citations"])
            print(
                f"  {code_id} ({info['name']}): "
                f"{len(info['citations'])} studies - "
                f"{sorted(info['citations'])}"
            )

    if not findings:
        print("\nNo LaTeX files found. Using default study IDs.")
        all_studies = (
            {f"S{i:02d}" for i in range(1, 19)}
            | {f"R{i:02d}" for i in range(1, 11)}
        )

    # ------------------------------------------------------------------
    # 2. Quality assessments
    # ------------------------------------------------------------------
    quality_data: Dict[str, StudyInfo] = {}

    if not quality_file.exists():
        print(f"\n{'=' * 70}")
        print("Generating quality assessment template ...")
        generate_quality_assessment_template(quality_file, sorted(all_studies))
        print(f"\nPlease fill in: {quality_file}")
        print("Then re-run this script to generate the Evidence Profile.")
    else:
        print(f"\nLoading quality assessments from: {quality_file}")
        quality_data = load_quality_assessments(quality_file)
        print(f"  Loaded {len(quality_data)} study assessments")
        print_quality_summary(quality_data)

    # ------------------------------------------------------------------
    # 3. Generate outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(exist_ok=True)

    if quality_data:
        print(f"\n{'=' * 70}")
        print("Generating LaTeX quality tables ...")
        generate_latex_quality_tables(quality_data, output_dir)

    if findings:
        if quality_data:
            print(f"\n{'=' * 70}")
            print("Generating CERQual templates ...")
            generate_evidence_profile_template(
                findings, quality_data,
                output_dir / "evidence_profile_template.csv",
            )

            generate_soqf_template(
                findings,
                output_dir / "soqf_template.csv",
            )

            print(f"\n{'=' * 70}")
            print("Generating LaTeX certainty-of-evidence tables ...")
            generate_latex_evidence_tables(findings, quality_data, output_dir)

    generate_summary_report(
        findings, all_studies,
        output_dir / "cerqual_summary.txt",
    )

    print(f"\n{'=' * 70}")
    print("NEXT STEPS:")
    print("1. Review the extracted citations in the summary report")
    print("2. Review the generated LaTeX tables for your manuscript")
    print("3. Fill in the Evidence Profile template with manual CERQual assessments")
    print("4. Create the SoQF table based on your assessments")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()