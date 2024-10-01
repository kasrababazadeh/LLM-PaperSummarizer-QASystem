import streamlit as st
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_from_chain_type
from langchain.llms import LLMChain
import os

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([""], embeddings)

# Create a QA chain
qa_chain = load_qa_with_sources_from_chain_type(
    "stuff",
    llm=LLMChain(llm=None),
    retriever=vectorstore.as_retriever(),
    sources=[],
)

def parse_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    num_pages = pdf_reader.numPages
    text = ''
    for page in range(num_pages):
        page_obj = pdf_reader.getPage(page)
        text += page_obj.extractText()
    pdf_file_obj.close()
    return text

def summarize_text(text, max_tokens=1000):
    # Split the text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    # Add chunks to the vector store
    vectorstore.add_texts(chunks)
    
    # Update the QA chain with new sources
    qa_chain.sources = [{"page_content": chunk} for chunk in chunks]
    
    # Generate a summary
    prompt = f"Summarize the content of the research paper in {max_tokens} tokens."
    result = qa_chain.run(prompt=prompt)
    
    return result

def answer_query(query):
    prompt = f"Answer the following query about the research paper: '{query}'"
    result = qa_chain.run(prompt=prompt)
    return result

st.title("Research Paper Summarizer and Q&A System")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
    st.write(file_details)
    
    # Save the uploaded file temporarily
    temp_file = os.path.join(os.getcwd(), uploaded_file.name)
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Parse the PDF
    text = parse_pdf(temp_file)
    
    # Summarize the content
    summary = summarize_text(text)
    
    st.subheader("Summary:")
    st.write(summary)
    
    # Remove the temporary file
    os.remove(temp_file)
    
    # Q&A session
    st.subheader("Ask a question about the paper:")
    query = st.text_input("")
    if st.button("Submit"):
        answer = answer_query(query)
        st.write(answer)
