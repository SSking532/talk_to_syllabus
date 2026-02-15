import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_pipeline import create_vector_store, ask_question

# ---------- Page Config ----------
st.set_page_config(
    page_title="SyllabAI",
    page_icon="📘",
    layout="wide"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>

body {
    background-color: #F5F7FA;
}

/* Header Styling */
.app-header {
    font-size: 45px;
    font-weight: bold;
    color: #2C3E50;
}

.tagline {
    font-size: 18px;
    color: #7F8C8D;
}

/* Chat Bubble User */
.user-bubble {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Chat Bubble AI */
.ai-bubble {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #3498DB;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

/* Upload Card */
.upload-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col1, col2 = st.columns([1, 8])

with col1:
    st.markdown("## 📘🤖")

with col2:
    st.markdown('<div class="app-header">SyllabAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Ask. Understand. Excel.</div>', unsafe_allow_html=True)

st.divider()

# ---------- Sidebar ----------
with st.sidebar:

    st.header("⚙️ Control Panel")

    st.markdown("""
    Upload your syllabus and start chatting with AI.
    """)

    uploaded_file = st.file_uploader("📄 Upload Syllabus", type="pdf")

    if uploaded_file:
        with st.spinner("Processing syllabus..."):
            text = extract_text_from_pdf(uploaded_file)
            create_vector_store(text)

        st.success("✅ Syllabus Loaded")

    st.divider()

    st.info("""
    💡 Try asking:
    - What topics are in Unit 2?
    - What are prerequisites?
    - Explain syllabus summary
    """)

# ---------- Welcome Banner ----------
st.markdown("""
### 👋 Welcome to SyllabAI

Upload your syllabus and ask questions instantly.  
Your AI academic assistant is ready!
""")

st.divider()

# ---------- Chat Section ----------
st.subheader("💬 Ask Your Question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Type your question here...")

if question:
    with st.spinner("🤖 Generating Answer..."):
        answer = ask_question(question)

    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("ai", answer))

# ---------- Display Chat ----------
for role, message in st.session_state.chat_history:

    if role == "user":
        st.markdown(f"""
        <div class="user-bubble">
        🧑 <b>You:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="ai-bubble">
        🤖 <b>SyllabAI:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)

# ---------- Footer ----------
st.divider()

st.markdown("""
<center>
Made with ❤️ for your assistance  
<b>SyllabAI – Your Academic Copilot</b>
</center>
""", unsafe_allow_html=True)
