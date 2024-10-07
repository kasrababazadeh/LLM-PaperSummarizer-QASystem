# Research Paper Summarizer and Q&A System

This project is an NLP-based system that summarizes research papers and provides an interactive Q&A feature. It uses **Hugging Face models**, **Langchain**, and **FAISS** for natural language processing, embeddings, and efficient vector-based retrieval. The system is built with **Streamlit** to create a user-friendly web interface for uploading PDF research papers and interacting with the summarizer and Q&A functionalities.

## Features

- **PDF Research Paper Summarization**: Automatically extract and summarize the content of research papers uploaded in PDF format.
- **Interactive Q&A**: Ask questions about the research paper, and get answers based on the summarized content.
- **Efficient Retrieval**: Powered by FAISS and Hugging Face sentence embeddings for fast and accurate text processing.

## Technologies Used

- [Streamlit](https://streamlit.io/) - A framework to build web applications in Python.
- [Langchain](https://github.com/hwchase17/langchain) - Provides utilities for working with Large Language Models (LLMs) and text.
- [Hugging Face](https://huggingface.co/) - Pretrained transformer models used for embedding research papers.
- [FAISS](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering.
- [PyPDF2](https://github.com/py-pdf/pypdf2) - A library to read and extract text from PDF files.

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `pip` (Python package installer)

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/kasrababazadeh/NLP-PaperSummarizer-QASystem.git
   cd repository-name

2. **Install dependencies**

   ```bash
   pip install streamlit PyPDF2 langchain faiss-cpu sentence-transformers

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py

## Usage

1. **Upload a PDF File**: Use the file uploader to choose a research paper in PDF format.
   
2. **Summary Generation**: Once the PDF is uploaded, the app will extract text from the file and summarize the content. The summary will appear under the "Summary" section.

3. **Ask a Question**: You can type a question in the Q&A input field about the research paper, and the system will provide an answer based on the content.

## Code Overview

- **`parse_pdf(file_path)`**: Extracts text from a PDF file.
- **`summarize_text(text, max_tokens)`**: Splits the extracted text into chunks, adds them to the FAISS vector store, and generates a summary using the Langchain-based QA chain.
- **`answer_query(query)`**: Handles user queries, retrieves relevant chunks from the vector store, and generates answers.
- **Streamlit App**: Provides the front-end interface for file upload, text summarization, and interactive Q&A.

## Example Workflow

1. **Step 1**: Upload a research paper in PDF format.
2. **Step 2**: The system summarizes the paper.
3. **Step 3**: Ask questions like:
   - "What is the main contribution of the paper?"
   - "Explain the methodology used in the research."
   - "What are the key findings?"

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
