"""
Document Processor - Handles loading and chunking documents
"""
import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def load_text_file(self, file_path: str) -> str:
        """Load text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_documents_from_directory(self, directory: str) -> List[str]:
        """Load all documents from a directory."""
        documents = []
        if not os.path.exists(directory):
            return documents
            
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if filename.endswith('.pdf'):
                    documents.append(self.load_pdf(file_path))
                elif filename.endswith('.txt'):
                    documents.append(self.load_text_file(file_path))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return documents

    def chunk_text(self, text: str) -> List[str]:
        """Split a single text into smaller chunks."""
        return self.text_splitter.split_text(text)

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into smaller chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks

