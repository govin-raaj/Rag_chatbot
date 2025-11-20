from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
import tempfile

class process_data:
    def __init__(self, file):
        self.file = file

    def process_document(self):
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, self.file.filename)
    
        try:
        # Save file temporarily
            self.file.save(temp_path)
            
            # Process based on file type
            if self.file.filename.endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
            elif self.file.filename.endswith('.txt'):
                loader = TextLoader(temp_path)
                documents = loader.load()
            else:
                raise ValueError("Unsupported file type")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            text_chunks = text_splitter.split_documents(documents)
            
            return text_chunks
        
        finally:
        # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)


    