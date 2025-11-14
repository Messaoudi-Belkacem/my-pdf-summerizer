import streamlit as st
from transformers import pipeline
import PyPDF2
import torch

# Page config
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📄 PDF Summarization Tool")
st.markdown("Upload a PDF document and get an AI-powered summary")

# Cache the model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    if st.button("Generate Summary"):
        with st.spinner("Processing your document..."):
            try:
                # Extract text
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    st.error("Could not extract text from PDF")
                else:
                    # Chunk text
                    words = text.split()
                    chunks = [' '.join(words[i:i+1000]) for i in range(0, len(words), 1000)]
                    
                    # Summarize
                    summaries = []
                    progress_bar = st.progress(0)
                    for idx, chunk in enumerate(chunks):
                        if len(chunk.split()) > 50:
                            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                            summaries.append(summary[0]['summary_text'])
                        progress_bar.progress((idx + 1) / len(chunks))
                    
                    final_summary = ' '.join(summaries)
                    
                    # Display results
                    st.success("✨ Summary Generated!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", len(words))
                    with col2:
                        st.metric("Summary Words", len(final_summary.split()))
                    with col3:
                        reduction = round((1 - len(final_summary.split()) / len(words)) * 100)
                        st.metric("Reduction", f"{reduction}%")
                    
                    st.subheader("Summary")
                    st.write(final_summary)
                    
                    # Download button
                    st.download_button(
                        label="Download Summary",
                        data=final_summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")