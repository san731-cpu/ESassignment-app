"""Local, extractive helpers for the clinical case review app."""

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Iterable

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError


@dataclass(frozen=True)
class ReviewQuestion:
    id: str
    title: str
    prompt: str
    keywords: tuple[str, ...]


QUESTIONS = (
    ReviewQuestion(
        "presenting-dilemma",
        "The Presenting Dilemma and Supervisory Need",
        'Why bring this case today? What specific clinical stuck point, ethical concern, or gut feeling prompted the clinician to prioritize this client right now?',
        ("stuck", "supervision", "supervisory", "ethical concern", "gut feeling", "uncertain", "dilemma", "consultation", "priority"),
    ),
    ReviewQuestion(
        "case-conceptualization",
        "Case Conceptualization and Theoretical Alignment",
        "How is the clinician making sense of the client's core patterns, and does the current treatment plan actually match that formulation?",
        ("formulation", "conceptualization", "core pattern", "treatment plan", "theoretical", "attachment", "cognitive", "psychodynamic", "intervention"),
    ),
    ReviewQuestion(
        "safety-risk",
        "Safety, Risk, and Duty-of-Care Management",
        "Have acute risks—suicidality, harm to others, abuse, or severe decompensation—been thoroughly assessed, documented, and addressed with a clear safety plan?",
        ("suicid", "self-harm", "harm to others", "homicid", "abuse", "safety plan", "risk assessment", "decompensation", "means", "intent", "protective factor"),
    ),
    ReviewQuestion(
        "alliance-ruptures",
        "The Therapeutic Alliance and Ruptures",
        "How is the working relationship functioning, are there subtle signs of withdrawal or tension, and how are power dynamics playing out in the room?",
        ("alliance", "therapeutic relationship", "rupture", "withdraw", "tension", "power dynamic", "trust", "engagement", "working relationship"),
    ),
    ReviewQuestion(
        "countertransference",
        "Countertransference and Clinician Reactivity",
        "What emotional reactions, personal triggers, urges to rescue, or feelings of frustration is this client stirring up in the clinician?",
        ("countertransference", "clinician reaction", "rescue", "frustrat", "trigger", "emotional reaction", "helpless", "anger", "boundary"),
    ),
    ReviewQuestion(
        "cultural-humility",
        "Cultural Humility and Blind Spots",
        "How are the intersecting identities, biases, and lived experiences of both the clinician and client shaping the work, and what might be going unexamined?",
        ("cultural", "identity", "identities", "bias", "lived experience", "race", "ethnicity", "gender", "intersection", "blind spot", "humility"),
    ),
    ReviewQuestion(
        "treatment-progress",
        "Treatment Progress and Clinical Outcomes",
        "What concrete evidence shows the client is moving toward their goals, or is the therapy drifting into supportive maintenance without real traction?",
        ("progress", "outcome", "goal", "maintenance", "symptom", "measure", "improvement", "setback", "traction", "treatment"),
    ),
    ReviewQuestion(
        "coordination",
        "Systemic and Interdisciplinary Coordination",
        "Who else is involved in the client's care—psychiatrists, primary care, family, vocational specialists, or community agencies—and where are the communication gaps?",
        ("psychiatr", "primary care", "family", "vocational", "community agency", "coordination", "interdisciplinary", "communication", "care team", "referral"),
    ),
    ReviewQuestion(
        "documentation-boundaries",
        "Documentation, Boundaries, and Ethical Compliance",
        "Are clinical notes up to date and defensible, and are clear frame boundaries being maintained around scheduling, contact, and scope of practice?",
        ("documentation", "clinical note", "boundary", "scheduling", "contact", "scope of practice", "ethical", "compliance", "record", "frame"),
    ),
    ReviewQuestion(
        "next-move",
        "The Clinician's Next Move",
        "Based on this review, what specific shift in intervention, pacing, or posture will the clinician take into the very next session?",
        ("next session", "next step", "intervention", "pacing", "posture", "adjust", "plan", "shift", "clinician will"),
    ),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Evidence:
    page: int
    text: str
    score: int


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract page text without writing the uploaded PDF to disk."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a PDF.")
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise FileNotDecryptedError("Encrypted PDFs are not supported.")
    return [
        (page_number, page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def _passages(pages: Iterable[tuple[int, str]], max_chars: int = 520) -> list[tuple[int, str]]:
    passages = []
    for page_number, page_text in pages:
        normalized = re.sub(r"-\s*\n\s*", "", page_text)
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
        for paragraph in re.split(r"\n\s*\n", normalized):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if not paragraph:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""
            for sentence in sentences:
                if current and len(current) + len(sentence) + 1 > max_chars:
                    passages.append((page_number, current.strip()))
                    current = ""
                if len(sentence) > max_chars:
                    if current:
                        passages.append((page_number, current.strip()))
                        current = ""
                    passages.extend(
                        (page_number, sentence[index:index + max_chars].strip())
                        for index in range(0, len(sentence), max_chars)
                    )
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                passages.append((page_number, current.strip()))
    return passages


def find_evidence(
    question: ReviewQuestion,
    pages: Iterable[tuple[int, str]],
    limit: int = 3,
) -> list[Evidence]:
    """Return matching source passages; this is lexical retrieval, not clinical analysis."""
    matches = []
    for page, passage in _passages(pages):
        passage_lower = passage.casefold()
        passage_tokens = set(_TOKEN_RE.findall(passage_lower))
        score = sum(
            1
            for keyword in question.keywords
            if (
                keyword.casefold() in passage_lower
                if " " in keyword
                else any(token.startswith(keyword.casefold()) for token in passage_tokens)
            )
        )
        if score:
            matches.append(Evidence(page=page, text=passage, score=score))
    matches.sort(key=lambda evidence: evidence.score, reverse=True)
    return matches[:limit]


def build_report(selected: Iterable[tuple[ReviewQuestion, list[Evidence]]]) -> str:
    """Format selected evidence as a portable Markdown review document."""
    sections = [
        "# Clinical Case Review — Source Evidence",
        "",
        "> This document contains locally matched excerpts, not AI-generated clinical analysis. "
        "Verify each excerpt in the source record. It is not a diagnosis, risk assessment, "
        "or substitute for clinical judgment or supervision.",
        "",
    ]
    for question, evidence_items in selected:
        sections.extend([f"## {question.title}", "", f"**Review prompt:** {question.prompt}", ""])
        if evidence_items:
            for evidence in evidence_items:
                sections.append(f"- **PDF page {evidence.page}:** “{evidence.text}”")
        else:
            sections.append("- No matching text was found by local keyword search. Review the PDF manually.")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"
