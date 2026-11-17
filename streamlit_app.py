import hashlib

import streamlit as st
from pypdf.errors import FileNotDecryptedError, PdfReadError

from review_core import (
    MAX_ANALYSIS_CHARS,
    QUESTIONS,
    CaseSummary,
    LocalAnalysisError,
    LocalModelUnavailable,
    analyze_selected_questions,
    build_report,
    check_local_model,
    extract_pdf_pages,
)


st.set_page_config(
    page_title="Clinical Case Review",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #172f3a; --muted: #647a80; --teal: #167c79; --pale: #eff8f6; }
    .stApp { background: #f7faf9; color: var(--ink); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); }
    p, label, button { font-family: 'DM Sans', sans-serif; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px; color: white;
        background: linear-gradient(115deg, #153a46 0%, #167c79 100%);
        margin: .4rem 0 1.6rem;
    }
    .hero h1 { color: white; font-size: 2.4rem; margin: .35rem 0; }
    .hero p { color: #d7edeb; max-width: 760px; margin-bottom: 0; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-weight: 700; font-size: .73rem; color: #a9ddcf; }
    .privacy-note {
        border: 1px solid #d5e8e2; background: #edf7f3; color: #264b4b;
        border-radius: 14px; padding: 1rem 1.1rem; margin: 0 0 1.4rem;
    }
    .section-label { color: var(--teal); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .75rem; }
    .stButton > button {
        border-radius: 12px; border: 1px solid #dce8e5; color: #183c43;
        background: white; min-height: 3.2rem; font-weight: 600; text-align: left;
    }
    .stButton > button:hover { border-color: #17847d; color: #126f6c; background: #f0faf7; }
    [data-testid="stSidebar"] { background: #f0f7f5; }
    .stDownloadButton > button {
        border-radius: 11px; background: #167c79; color: white; border: 0; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">A thoughtful second look</div>
      <h1>Clinical Case Review</h1>
      <p>Bring focused supervision questions to a case. Analyze the full record with a local model, review its cited evidence, and collect findings for consultation.</p>
    </div>
    <div class="privacy-note"><strong>Protect client privacy.</strong> De-identify records before uploading. PDF text is sent only to Ollama at 127.0.0.1 on the machine running this app; no cloud model API is used. If this app runs on a remote host, its local Ollama processes the PDF there. Verify all generated findings and citations; this tool is not an authoritative clinical assessment.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Local model setup")
    st.write("Install Ollama on the same machine as this app and download the default model:")
    st.code("ollama pull qwen2.5:7b")
    st.caption("Keep Ollama running at 127.0.0.1:11434. Set OLLAMA_MODEL to use another model already installed there.")
    try:
        local_model_name = check_local_model()
        st.success(f"Local model ready: {local_model_name}")
    except LocalModelUnavailable as exc:
        local_model_name = None
        st.warning(str(exc))

uploaded_pdf = st.file_uploader("Upload a de-identified PDF", type=["pdf"], accept_multiple_files=False, max_upload_size=50)

pages: list[tuple[int, str]] = []
if uploaded_pdf is None:
    st.session_state["document_hash"] = None
    st.session_state["selected_question_ids"] = set()
    st.session_state["case_summary"] = None
    st.info("Upload a PDF to analyze the full document and open the ten review prompts.")
else:
    pdf_bytes = uploaded_pdf.getvalue()
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("document_hash") != document_hash:
        st.session_state["document_hash"] = document_hash
        st.session_state["selected_question_ids"] = set()
        st.session_state["case_summary"] = None
    try:
        pages = extract_pdf_pages(pdf_bytes)
        if not any(text.strip() for _, text in pages):
            st.warning("No selectable text was found. This may be a scanned PDF; OCR is not available in this app.")
        else:
            if local_model_name:
                st.success(f"PDF ready · {len(pages)} pages · local model `{local_model_name}` is ready")
            else:
                st.warning(f"PDF ready · {len(pages)} pages. See the local model setup in the sidebar.")
            total_chars = sum(len(text) for _, text in pages)
            if total_chars > MAX_ANALYSIS_CHARS:
                st.warning(
                    f"This PDF has {total_chars:,} text characters, above the "
                    f"{MAX_ANALYSIS_CHARS:,}-character single-pass model limit. Split it before analysis."
                )
    except ValueError:
        st.error("This file does not appear to be a valid PDF. Choose a PDF and try again.")
    except (PdfReadError, FileNotDecryptedError) as exc:
        st.error(f"Could not read this PDF ({type(exc).__name__}). Check that it is not damaged or password-protected.")

if pages and any(text.strip() for _, text in pages):
    selected_ids = st.session_state.setdefault("selected_question_ids", set())
    st.markdown('<div class="section-label">Choose review questions</div>', unsafe_allow_html=True)
    st.caption("Select one or more prompts. The full PDF is analyzed once to generate the selected answers and a combined executive summary.")
    columns = st.columns(2, gap="medium")
    for index, question in enumerate(QUESTIONS):
        with columns[index % 2]:
            label = f"{index + 1:02d}  ·  {question.title}"
            is_selected = question.id in selected_ids
            if st.button(
                label,
                key=f"question-{question.id}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                if is_selected:
                    selected_ids.remove(question.id)
                else:
                    selected_ids.add(question.id)
                st.session_state["case_summary"] = None
                st.rerun()

    selected_questions = [question for question in QUESTIONS if question.id in selected_ids]
    if selected_questions:
        st.caption(f"{len(selected_questions)} question(s) selected")
    if st.button(
        "Generate Clinical Case Summary",
        type="primary",
        use_container_width=True,
        disabled=not selected_questions,
    ):
        try:
            with st.spinner("Analyzing the full PDF and generating the selected findings..."):
                st.session_state["case_summary"] = analyze_selected_questions(selected_questions, pages)
        except (LocalModelUnavailable, LocalAnalysisError) as exc:
            st.error(str(exc))

    case_summary: CaseSummary | None = st.session_state.get("case_summary")
    if case_summary:
        st.divider()
        st.subheader("Clinical Case Summary")
        st.markdown("### Executive summary")
        st.markdown(case_summary.executive_summary.answer)
        if case_summary.executive_summary.citations:
            for citation in case_summary.executive_summary.citations:
                st.markdown(f'- **Critical detail · PDF page {citation.page}** · "{citation.quote}"')
        else:
            st.info("The local model identified insufficient documented information for the executive summary.")

        st.markdown("### Question-specific findings")
        for question, result in case_summary.analyses:
            with st.container(border=True):
                st.markdown(f"#### {question.title}")
                st.markdown(f"_{question.prompt}_")
                st.markdown(result.answer)
                if result.citations:
                    st.markdown("**Verified, question-specific citations**")
                    for citation in result.citations:
                        st.markdown(f'- **PDF page {citation.page}** · "{citation.quote}"')
                else:
                    st.info("The local model identified insufficient relevant documentation for this question.")
        export = build_report(case_summary)
        st.download_button(
            "Download Clinical Case Summary",
            data=export.encode("utf-8"),
            file_name="clinical-case-summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if selected_ids and st.button("Clear selected questions", type="secondary"):
        st.session_state["selected_question_ids"] = set()
        st.session_state["case_summary"] = None
        st.rerun()

st.divider()
st.caption("Whole-document analysis by a local model · Verify every citation · Always use professional judgment and supervision")
