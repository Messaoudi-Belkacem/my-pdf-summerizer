from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, max_length=1024):
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text = extract_text_from_pdf(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if not text.strip():
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Split text into chunks if too long
        chunks = chunk_text(text, max_length=1024)
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:  # Only summarize if chunk is substantial
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        # Combine summaries
        final_summary = ' '.join(summaries)
        
        # If combined summary is still too long, summarize again
        if len(final_summary.split()) > 300:
            final_summary = summarizer(final_summary, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        
        return jsonify({
            'original_length': len(text.split()),
            'summary_length': len(final_summary.split()),
            'summary': final_summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)