# =============================================================
# app.py — Streamlit Web Interface for AI Code Reviewer
# Run with: streamlit run app/app.py
# =============================================================

import streamlit as st
import sys
import os
import markdown2
from datetime import datetime

# Add parent directory to path so we can import inference.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import get_reviewer

# =============================================================
# PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="AI Code Reviewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .review-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    language = st.selectbox(
        "Programming Language",
        ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust"],
        index=0,
        help="Select the language of your code snippet"
    )
    
    max_tokens = st.slider(
        "Review Length",
        min_value=128,
        max_value=1024,
        value=512,
        step=64,
        help="Maximum length of the generated review"
    )
    
    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
    This tool uses a **fine-tuned Qwen2.5-1.5B** model trained specifically 
    for code review tasks using **QLoRA** (4-bit quantization + LoRA adapters).
    
    It can detect:
    - 🐛 Bugs & logical errors
    - 🔒 Security vulnerabilities  
    - ⚡ Performance issues
    - 📚 Best practice violations
    """)
    
    st.divider()
    st.markdown("### 🧪 Sample Code")
    
    sample_codes = {
        "SQL Injection": '''def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)''',
        
        "Unclosed File": '''def read_file(filename):
    f = open(filename, 'r')
    data = f.read()
    return data''',
        
        "Bare Except": '''def process(data):
    try:
        result = compute(data)
    except:
        pass
    return result''',
    }
    
    selected_sample = st.selectbox("Load a sample:", ["None"] + list(sample_codes.keys()))


# =============================================================
# MAIN INTERFACE
# =============================================================

st.markdown('<div class="main-header">🔍 AI Code Reviewer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fine-tuned on Qwen2.5-1.5B using QLoRA • Specialized in code review</div>', unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Your Code")
    
    # Pre-fill with sample if selected
    default_code = ""
    if selected_sample != "None":
        default_code = sample_codes[selected_sample]
    
    code_input = st.text_area(
        "Paste your code here:",
        value=default_code,
        height=400,
        placeholder="def my_function():\n    # paste your code here...",
        label_visibility="collapsed"
    )
    
    # Character/line count
    if code_input:
        lines = len(code_input.split('\n'))
        chars = len(code_input)
        st.caption(f"📊 {lines} lines • {chars} characters")
    
    # Review button
    review_btn = st.button(
        "🔍 Review My Code",
        type="primary",
        use_container_width=True,
        disabled=not code_input.strip()
    )

with col2:
    st.subheader("📋 AI Review")
    
    if review_btn and code_input.strip():
        # Load model (cached after first load)
        with st.spinner("🤖 Analyzing your code... (first run loads the model ~30s)"):
            try:
                reviewer = get_reviewer()
                review = reviewer.review_code(
                    code=code_input,
                    language=language,
                    max_new_tokens=max_tokens
                )
                
                # Store in session state so it persists
                st.session_state["last_review"] = review
                st.session_state["last_code"] = code_input
                st.session_state["last_language"] = language
                st.session_state["review_time"] = datetime.now().strftime("%H:%M:%S")
                
            except Exception as e:
                st.error(f"❌ Error generating review: {str(e)}")
                st.info("Make sure the model is loaded correctly and you have enough RAM.")
    
    # Display review if available
    if "last_review" in st.session_state:
        review_text = st.session_state["last_review"]
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📖 Rendered", "📄 Raw Markdown"])
        
        with tab1:
            st.markdown(review_text)
        
        with tab2:
            st.code(review_text, language="markdown")
        
        st.caption(f"⏰ Generated at {st.session_state.get('review_time', '')}")
        
        # Download button
        st.download_button(
            label="⬇️ Download Review (.md)",
            data=f"# Code Review - {st.session_state.get('last_language', 'Code')}\n\n"
                 f"## Original Code\n```{st.session_state.get('last_language', '').lower()}\n"
                 f"{st.session_state.get('last_code', '')}\n```\n\n"
                 f"## Review\n{review_text}",
            file_name=f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    elif not review_btn:
        # Empty state
        st.markdown("""
        <div style="text-align:center; color:#999; padding:3rem">
            <div style="font-size:3rem">🤖</div>
            <div>Paste your code on the left and click <strong>Review My Code</strong></div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# HISTORY SECTION
# =============================================================

st.divider()
st.subheader("📚 Review History")

if "review_history" not in st.session_state:
    st.session_state["review_history"] = []

# Add current review to history
if review_btn and "last_review" in st.session_state:
    entry = {
        "time": st.session_state["review_time"],
        "language": st.session_state["last_language"],
        "code_preview": st.session_state["last_code"][:80] + "...",
        "review": st.session_state["last_review"]
    }
    # Only add if not duplicate
    if not st.session_state["review_history"] or \
       st.session_state["review_history"][-1]["review"] != entry["review"]:
        st.session_state["review_history"].append(entry)

if st.session_state["review_history"]:
    for i, entry in enumerate(reversed(st.session_state["review_history"][-5:])):
        with st.expander(f"[{entry['time']}] {entry['language']} — {entry['code_preview']}"):
            st.markdown(entry["review"])
else:
    st.caption("Your review history will appear here.")
