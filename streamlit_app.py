import streamlit as st
from pypdf.errors import FileNotDecryptedError, PdfReadError

from review_core import QUESTIONS, build_report, extract_pdf_pages, find_evidence


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
      <p>Bring focused supervision questions to a case. Find relevant passages in a PDF, then collect the evidence you want to take into consultation.</p>
    </div>
    <div class="privacy-note"><strong>Protect client privacy.</strong> De-identify the record before uploading. The PDF is processed in this app session and is not sent to an AI service or saved by this app. Verify every excerpt against the original record; this tool does not provide clinical assessment or advice.</div>
    """,
    unsafe_allow_html=True,
)

uploaded_pdf = st.file_uploader("Upload a de-identified PDF", type=["pdf"], accept_multiple_files=False, max_upload_size=50)

pages: list[tuple[int, str]] = []
if uploaded_pdf is None:
    st.info("Upload a PDF to search its text locally and open the ten review prompts.")
else:
    pdf_bytes = uploaded_pdf.getvalue()
    try:
        pages = extract_pdf_pages(pdf_bytes)
        st.success(f"PDF ready · {len(pages)} pages · processed locally")
        if not any(text.strip() for _, text in pages):
            st.warning("No selectable text was found. This may be a scanned PDF; OCR is not available in this app.")
    except ValueError:
        st.error("This file does not appear to be a valid PDF. Choose a PDF and try again.")
    except (PdfReadError, FileNotDecryptedError) as exc:
        st.error(f"Could not read this PDF ({type(exc).__name__}). Check that it is not damaged or password-protected.")

if pages and any(text.strip() for _, text in pages):
    selected_ids = st.session_state.setdefault("selected_question_ids", set())
    st.markdown('<div class="section-label">Choose a review question</div>', unsafe_allow_html=True)
    st.caption("Each button finds a few matching excerpts; it does not interpret them or fill in information that is absent.")
    columns = st.columns(2, gap="medium")
    for index, question in enumerate(QUESTIONS):
        with columns[index % 2]:
            label = f"{index + 1:02d}  ·  {question.title}"
            if st.button(label, key=f"question-{question.id}", use_container_width=True):
                selected_ids.add(question.id)

    selected_questions = [
        (question, find_evidence(question, pages))
        for question in QUESTIONS
        if question.id in selected_ids
    ]
    if selected_questions:
        st.divider()
        st.subheader("Requested findings")
        for question, evidence_items in selected_questions:
            with st.container(border=True):
                st.markdown(f"#### {question.title}")
                st.markdown(f"_{question.prompt}_")
                if evidence_items:
                    st.caption(f"Local keyword matches · {len(evidence_items)} excerpt(s)")
                    for evidence in evidence_items:
                        st.markdown(f"**PDF page {evidence.page}**")
                        st.write(f"“{evidence.text}”")
                else:
                    st.info("No matching text was found by local keyword search. Review this question manually in the source PDF.")
        export = build_report(selected_questions)
        st.download_button(
            "Download requested findings",
            data=export.encode("utf-8"),
            file_name="clinical-case-review.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("Clear selected questions", type="secondary"):
            st.session_state["selected_question_ids"] = set()
            st.rerun()

st.divider()
st.caption("Local evidence search only · No AI analysis is configured · Always use professional judgment and supervision")
