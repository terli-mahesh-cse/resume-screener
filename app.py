import streamlit as st
import sys
import os

# Add app folder to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

from parser import DocumentParser
from matcher import SemanticMatcher
# Page config
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# Title
st.title("📄 AI Resume Screener")
st.markdown("Upload a resume and paste a job description to get a match score.")

# Initialize
@st.cache_resource
def get_matcher():
    return SemanticMatcher()

matcher = get_matcher()
parser = DocumentParser()

# Sidebar
st.sidebar.header("Settings")
st.sidebar.info("Model: all-MiniLM-L6-v2")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader("Choose PDF, DOCX, or TXT", type=['pdf', 'docx', 'txt'])
    
    resume_text = ""
    if uploaded_file:
        # Save temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        resume_text = parser.parse(temp_path)
        os.remove(temp_path)
        
        st.success(f"✅ Parsed {uploaded_file.name}")
        with st.expander("Preview Resume Text"):
            st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

with col2:
    st.subheader("📝 Job Description")
    job_desc = st.text_area("Paste job description here", height=300)

# Match button
if st.button("🔍 Calculate Match", type="primary", use_container_width=True):
    if not resume_text or not job_desc:
        st.error("⚠️ Please upload a resume AND enter a job description!")
    else:
        with st.spinner("Analyzing... This takes ~10 seconds"):
            result = matcher.match(resume_text, job_desc)
        
        # Results
        st.divider()
        st.subheader("📊 Results")
        
        score = result['overall_match']
        color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        
        col_score, col_info = st.columns([1, 2])
        
        with col_score:
            st.metric("Match Score", f"{score}%")
            if score >= 70:
                st.success(f"{color} Strong Match!")
            elif score >= 50:
                st.warning(f"{color} Moderate Match")
            else:
                st.error(f"{color} Weak Match")
        
        with col_info:
            st.write(f"**Resume chunks analyzed:** {result['total_resume_chunks']}")
            st.write(f"**JD chunks analyzed:** {result['total_jd_chunks']}")
            st.write(f"**Matched sections:** {len(result['matched_sections'])}")
        
        # Matched sections
        if result['matched_sections']:
            st.subheader("🔗 Top Matched Sections")
            for i, match in enumerate(result['matched_sections'], 1):
                with st.expander(f"Match #{i} — Score: {match['score']}%"):
                    st.markdown(f"**Resume:** {match['resume_text']}")
                    st.markdown(f"**Job Description:** {match['jd_text']}")

# Footer
st.divider()
st.caption("Built with Python 3.11 + Streamlit + Sentence-BERT")