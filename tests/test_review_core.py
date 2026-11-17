from io import BytesIO
import unittest

from pypdf import PdfWriter
from pypdf.errors import FileNotDecryptedError

from review_core import QUESTIONS, build_report, extract_pdf_pages, find_evidence


class ReviewCoreTests(unittest.TestCase):
    def test_all_ten_review_prompts_are_preserved(self):
        self.assertEqual(len(QUESTIONS), 10)
        self.assertIn("suicidality", QUESTIONS[2].prompt)
        self.assertIn("blind spots", QUESTIONS[5].title.lower())
        self.assertIn("very next session", QUESTIONS[9].prompt)

    def test_evidence_search_returns_matching_page_and_source_text(self):
        pages = [
            (1, "The client described feeling hopeful about work."),
            (2, "A safety plan was reviewed after the client reported suicidal thoughts."),
        ]

        evidence = find_evidence(QUESTIONS[2], pages)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].page, 2)
        self.assertIn("safety plan", evidence[0].text)

    def test_no_match_does_not_invent_evidence_and_report_discloses_limit(self):
        evidence = find_evidence(QUESTIONS[0], [(1, "A session was held on Tuesday.")])

        self.assertEqual(evidence, [])
        report = build_report([(QUESTIONS[0], evidence)])
        self.assertIn("No matching text was found by local keyword search", report)
        self.assertIn("not AI-generated clinical analysis", report)

    def test_report_contains_requested_prompt_and_cited_excerpt(self):
        evidence = find_evidence(QUESTIONS[7], [(4, "The psychiatrist coordinated with primary care.")])

        report = build_report([(QUESTIONS[7], evidence)])

        self.assertIn(QUESTIONS[7].prompt, report)
        self.assertIn("PDF page 4", report)
        self.assertIn("The psychiatrist coordinated with primary care.", report)

    def test_pdf_pages_are_extracted_in_memory_and_encrypted_pdfs_are_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf = BytesIO()
        writer.write(pdf)

        self.assertEqual(extract_pdf_pages(pdf.getvalue()), [(1, "")])
        with self.assertRaises(ValueError):
            extract_pdf_pages(b"not a pdf")

        writer.encrypt("test-password")
        encrypted_pdf = BytesIO()
        writer.write(encrypted_pdf)
        with self.assertRaises(FileNotDecryptedError):
            extract_pdf_pages(encrypted_pdf.getvalue())


if __name__ == "__main__":
    unittest.main()
