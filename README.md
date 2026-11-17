# Clinical Case Review

An in-session PDF evidence finder for ten clinical supervision prompts. Upload a de-identified PDF, choose questions, review the matching source excerpts, and download the selected findings as Markdown.

## Privacy and limitations

- PDF text extraction and keyword matching run locally in the Streamlit app process. This app does not send records to an AI service or write their contents to disk.
- Uploaded files remain in memory for the active Streamlit session. Closing the session clears the app's in-memory upload and selected questions; hosting infrastructure may have its own session and operational policies.
- The app has no AI/model integration. It surfaces lexical matches from the source PDF, not synthesized answers, clinical recommendations, or a safety assessment. Scanned PDFs need OCR, which is not included.
- De-identify records before upload and verify every excerpt against the original. This tool is not a substitute for clinical judgment, supervision, or professional obligations.

## Run locally

Prerequisite: install `uv`.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync dependencies: `uv sync`
2. Start the app: `uv run streamlit run streamlit_app.py`
3. Run focused tests: `uv run python -m unittest discover -s tests`
