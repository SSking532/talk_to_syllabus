
import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_pipeline import create_vector_store, ask_question


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="SyllabAI",
    page_icon="📘",
    layout="wide"
)


# --------------------------------
# SESSION STATE
# --------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""


# --------------------------------
# CUSTOM STYLING
# --------------------------------

st.markdown("""
<style>

body {
    background-color: #F5F7FA;
}

.app-header {
    font-size: 45px;
    font-weight: bold;
    color: #2C3E50;
}

.tagline {
    font-size: 18px;
    color: #7F8C8D;
}

.user-bubble {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 8px;
}

.ai-bubble {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #3498DB;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

.upload-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

col1, col2 = st.columns([1, 8])

with col1:
    st.markdown("## 📘🤖")

with col2:
    st.markdown(
        '<div class="app-header">SyllabAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tagline">Ask. Understand. Excel.</div>',
        unsafe_allow_html=True
    )

st.divider()


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.header("⚙️ Control Panel")

    st.markdown("""
    Upload your syllabus and start chatting with AI.
    """)

    uploaded_file = st.file_uploader(
        "📄 Upload Syllabus",
        type="pdf"
    )

    if uploaded_file is not None:

        file_id = (
            uploaded_file.name,
            uploaded_file.size
        )

        # Process only when a new file is uploaded
        if st.session_state.processed_file != file_id:

            with st.spinner("Processing syllabus..."):

                try:
                    text = extract_text_from_pdf(uploaded_file)

                    if not text.strip():
                        st.error("No readable text found in the PDF.")

                    else:
                        create_vector_store(text)

                        st.session_state.extracted_text = text
                        st.session_state.processed_file = file_id
                        st.session_state.chat_history = []

                        st.success("✅ Syllabus Loaded Successfully!")

                except Exception as e:
                    st.error(f"PDF processing failed: {str(e)}")

        else:
            st.success("✅ Syllabus Loaded")

    # Extraction verification
    if st.session_state.extracted_text:

        st.divider()

        st.subheader("📊 PDF Verification")

        st.metric(
            "Extracted Characters",
            f"{len(st.session_state.extracted_text):,}"
        )

        with st.expander("View Extracted PDF Text"):

            st.text_area(
                "Extracted Text Preview",
                st.session_state.extracted_text[:5000],
                height=300
            )

            st.caption(
                "Preview shows the first 5,000 characters "
                "of the extracted PDF."
            )

        if st.button("🔄 Reset Syllabus", use_container_width=True):

            st.session_state.processed_file = None
            st.session_state.extracted_text = ""
            st.session_state.chat_history = []

            st.rerun()

    st.divider()

    st.info("""
    💡 Try asking:

    - What topics are in Unit 2?
    - Which chapter covers electricity?
    - Is photosynthesis included?
    - Summarize Unit 10.
    """)


# --------------------------------
# WELCOME BANNER
# --------------------------------

st.markdown("""
### 👋 Welcome to SyllabAI

Upload your syllabus and ask questions instantly.

Your AI academic assistant is ready!
""")

st.divider()


# --------------------------------
# CHAT SECTION
# --------------------------------

st.subheader("💬 Ask Your Question")

question = st.text_input(
    "Type your question here...",
    key="question_input",
    placeholder="Example: What topics are covered in Unit 2?"
)

ask_button = st.button(
    "Ask SyllabAI 🤖",
    use_container_width=False
)

if ask_button:

    if not st.session_state.extracted_text:
        st.warning("Please upload a syllabus PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("🤖 Generating Answer..."):

            try:
                answer = ask_question(question.strip())

                st.session_state.chat_history.append(
                    ("user", question.strip())
                )

                st.session_state.chat_history.append(
                    ("ai", answer)
                )

            except Exception as e:
                st.error(f"Unable to generate answer: {str(e)}")


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for role, message in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"""
            <div class="user-bubble">
                🧑 <b>You:</b><br>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="ai-bubble">
                🤖 <b>SyllabAI:</b><br>
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown("""
<center>
Made with ❤️ for your assistance

<b>SyllabAI – Your Academic Copilot</b>
</center>
""", unsafe_allow_html=True)
