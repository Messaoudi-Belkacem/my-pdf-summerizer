import gradio as gr
from transformers import pipeline
import PyPDF2
import io

# Load model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_pdf(pdf_file):
    try:
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        if not text.strip():
            return "Error: Could not extract text from PDF"
        
        # Chunk and summarize
        words = text.split()
        chunks = [' '.join(words[i:i+1000]) for i in range(0, len(words), 1000)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        final_summary = ' '.join(summaries)
        
        return f"**Summary:**\n\n{final_summary}\n\n**Stats:** {len(words)} words → {len(final_summary.split())} words"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(label="Upload PDF", file_types=[".pdf"]),
    outputs=gr.Textbox(label="Summary", lines=10),
    title="📄 PDF Summarization Tool",
    description="Upload a PDF document and get an AI-powered summary using BART",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()