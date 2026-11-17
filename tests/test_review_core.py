from io import BytesIO
import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.errors import FileNotDecryptedError

from review_core import (
    MAX_ANALYSIS_CHARS,
    QUESTIONS,
    AnalysisResult,
    CaseSummary,
    Citation,
    LocalAnalysisError,
    LocalModelUnavailable,
    analyze_selected_questions,
    build_report,
    check_local_model,
    extract_pdf_pages,
    validate_analysis_response,
)


class FakeResponse:
    def __init__(self, value):
        self.value = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.value


class ReviewCoreTests(unittest.TestCase):
    def test_all_ten_review_prompts_are_preserved(self):
        self.assertEqual(len(QUESTIONS), 10)
        self.assertIn("suicidality", QUESTIONS[2].prompt)
        self.assertIn("blind spots", QUESTIONS[5].title.lower())
        self.assertIn("very next session", QUESTIONS[9].prompt)
        self.assertFalse(hasattr(QUESTIONS[0], "keywords"))

    def test_selected_questions_and_executive_summary_use_one_full_document_request(self):
        pages = [
            (1, "The client described feeling hopeful about work."),
            (2, "A safety plan was reviewed with the client after the crisis."),
        ]
        questions = [QUESTIONS[2], QUESTIONS[7]]
        model_response = {
            "executive_summary": {
                "answer": "The record notes a recent crisis and a safety plan. [p. 2]",
                "citations": [{"page": 2, "quote": "A safety plan was reviewed with the client"}],
            },
            "analyses": [
                {
                    "question_id": QUESTIONS[2].id,
                    "answer": "A safety plan followed the recent crisis. [p. 2]",
                    "citations": [{"page": 2, "quote": "A safety plan was reviewed with the client"}],
                },
                {
                    "question_id": QUESTIONS[7].id,
                    "answer": "Care coordination is not documented in the PDF.",
                    "citations": [],
                },
            ],
        }
        with patch(
            "review_core.urlopen",
            side_effect=[
                FakeResponse({"models": [{"name": "qwen2.5:7b"}]}),
                FakeResponse({"message": {"content": json.dumps(model_response)}}),
            ],
        ) as mocked_urlopen:
            summary = analyze_selected_questions(questions, pages)

        self.assertIsInstance(summary, CaseSummary)
        self.assertEqual([question.id for question, _ in summary.analyses], [q.id for q in questions])
        self.assertEqual(summary.executive_summary.citations[0].page, 2)
        self.assertEqual(mocked_urlopen.call_count, 2)
        request = mocked_urlopen.call_args_list[1].args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        payload = json.loads(request.data)
        full_prompt = payload["messages"][1]["content"]
        self.assertIn("The client described feeling hopeful about work.", full_prompt)
        self.assertIn("A safety plan was reviewed with the client after the crisis.", full_prompt)
        self.assertIn(QUESTIONS[2].prompt, full_prompt)
        self.assertIn(QUESTIONS[7].prompt, full_prompt)
        self.assertIn("Units will be calculated.", full_prompt)
        self.assertIn("Do not use it as evidence of actual progress or symptoms.", full_prompt)
        self.assertIn("concrete next steps", full_prompt)
        self.assertEqual(payload["stream"], False)

    def test_unavailable_local_model_has_actionable_setup_error(self):
        with patch("review_core.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(LocalModelUnavailable, "Install and start Ollama"):
                check_local_model()

    def test_missing_model_reports_pull_command(self):
        with patch("review_core.urlopen", return_value=FakeResponse({"models": []})):
            with self.assertRaisesRegex(LocalModelUnavailable, "ollama pull qwen2.5:7b"):
                check_local_model()

    def test_missing_question_analysis_does_not_produce_a_partial_summary(self):
        incomplete_response = {
            "executive_summary": {
                "answer": "There is insufficient information to answer.",
                "citations": [],
            },
            "analyses": [],
        }
        with patch(
            "review_core.urlopen",
            side_effect=[
                FakeResponse({"models": [{"name": "qwen2.5:7b"}]}),
                FakeResponse({"message": {"content": json.dumps(incomplete_response)}}),
            ],
        ):
            with self.assertRaisesRegex(LocalAnalysisError, "did not answer every selected question"):
                analyze_selected_questions([QUESTIONS[0]], [(1, "A clinical note.")])

    def test_citation_must_match_source_page_and_inline_page_reference(self):
        pages = [(1, "The clinician discussed a safety plan with the client.")]
        valid = {
            "answer": "The clinician discussed a safety plan. [p. 1]",
            "citations": [{"page": 1, "quote": "discussed a safety plan with the client"}],
        }
        result = validate_analysis_response(json.dumps(valid), pages)
        self.assertEqual(result.citations[0].page, 1)

        invalid_quote = {
            "answer": "The clinician discussed a safety plan. [p. 1]",
            "citations": [{"page": 1, "quote": "The client denied suicidal thoughts"}],
        }
        with self.assertRaisesRegex(LocalAnalysisError, "could not be verified"):
            validate_analysis_response(json.dumps(invalid_quote), pages)

        invalid_page = {
            "answer": "A safety plan is documented. [p. 2]",
            "citations": [{"page": 2, "quote": "discussed a safety plan"}],
        }
        with self.assertRaisesRegex(LocalAnalysisError, "could not be verified"):
            validate_analysis_response(json.dumps(invalid_page), pages)

    def test_answer_without_evidence_must_explicitly_state_the_gap(self):
        response = {
            "answer": "There is insufficient information to answer this question from the PDF.",
            "citations": [],
        }
        self.assertEqual(
            validate_analysis_response(json.dumps(response), [(1, "An unrelated note.")]).citations,
            (),
        )
        response["answer"] = "The client needs a specific treatment change."
        with self.assertRaisesRegex(LocalAnalysisError, "no source citations"):
            validate_analysis_response(json.dumps(response), [(1, "An unrelated note.")])

    def test_oversized_document_is_not_truncated_or_sent(self):
        pages = [(1, "x" * (MAX_ANALYSIS_CHARS + 1))]
        with self.assertRaisesRegex(LocalAnalysisError, "no text was truncated or analyzed"):
            analyze_selected_questions([QUESTIONS[0]], pages)

    def test_report_contains_executive_summary_and_question_citations(self):
        summary = CaseSummary(
            executive_summary=AnalysisResult(
                "The record describes coordinated follow-up. [p. 4]",
                (Citation(4, "The psychiatrist coordinated with primary care."),),
            ),
            analyses=(
                (
                    QUESTIONS[7],
                    AnalysisResult(
                        "Care coordination is documented. [p. 4]",
                        (Citation(4, "The psychiatrist coordinated with primary care."),),
                    ),
                ),
            ),
        )
        report = build_report(summary)

        self.assertIn("# Clinical Case Summary", report)
        self.assertIn("## Executive summary", report)
        self.assertIn("## Selected question analyses", report)
        self.assertIn(QUESTIONS[7].prompt, report)
        self.assertIn("The record describes coordinated follow-up.", report)
        self.assertIn('- **[p. 4]** "The psychiatrist coordinated with primary care."', report)
        self.assertIn("verify every quote and interpretation", report)

    def test_pdf_pages_are_extracted_in_memory_and_encrypted_pdfs_are_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf = BytesIO()
        writer.write(pdf)

        self.assertEqual(extract_pdf_pages(pdf.getvalue()), [(1, "")])
        with self.assertRaises(ValueError):
            extract_pdf_pages(b"not a pdf")

        encrypted_writer = PdfWriter()
        encrypted_writer.add_blank_page(width=72, height=72)
        encrypted_writer.encrypt("test-password")
        encrypted_pdf = BytesIO()
        encrypted_writer.write(encrypted_pdf)
        with self.assertRaises(FileNotDecryptedError):
            extract_pdf_pages(encrypted_pdf.getvalue())


if __name__ == "__main__":
    unittest.main()
