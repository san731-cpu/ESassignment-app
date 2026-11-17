# Clinical Case Review

An in-session PDF review app for ten clinical supervision prompts. Select one or more questions, analyze the full document once, review a combined Clinical Case Summary with an executive summary and question-specific citations, then download it as Markdown.

## Local model and privacy

- The app extracts text in memory and sends the full page-labeled text only to Ollama at `127.0.0.1:11434`. It does not use a cloud model API, write uploaded content to disk, or log document text.
- Ollama must run on the same machine as the Streamlit app. If you host Streamlit remotely, the PDF is uploaded to that host and analyzed by its local Ollama; run the app on the clinician's machine if the record must stay there.
- Selected answers remain in the active app session's memory. The Ollama installation may have its own operational behavior and configuration; review that separately.
- Citations are checked against the extracted source page and exact quote, but the model's interpretation must still be verified against the original. This is not a diagnosis, authoritative risk assessment, or substitute for clinical judgment or supervision.

## Install and run

1. Install [Ollama](https://ollama.com/) on the same machine as this app and download the default model:

   ```powershell
   ollama pull qwen2.5:7b
   ```

2. Keep Ollama running at `127.0.0.1:11434`. The app shows an actionable unavailable-model message if the service or model is missing.
3. Install Python dependencies and start Streamlit:

   ```powershell
   uv sync
   uv run streamlit run streamlit_app.py
   ```

The default model is `qwen2.5:7b`. To use another model already installed in Ollama, set `OLLAMA_MODEL` before starting the app (for example, `$env:OLLAMA_MODEL = "llama3.1:8b"` in PowerShell).

The entire extracted document is passed for each selected question. To avoid silent truncation, analysis refuses documents over 75,000 extracted text characters; split a larger PDF before review. Scanned PDFs require OCR, which is not included.

Run focused tests with `uv run python -m unittest discover -s tests`.
