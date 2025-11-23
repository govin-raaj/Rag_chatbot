from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from typing import List
from langchain.schema import Document
from app.logger import logging

class ProcessData:
    """Process a saved file on disk and return a list of text chunks (langchain Documents)."""

    SUPPORTED_EXT = (".pdf", ".txt")

    def __init__(self, file_path: str):
        logging.info(f"Initializing ProcessData for file: {file_path}")
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string path to an existing file")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
        self.file_path = file_path
        self.filename = os.path.basename(file_path)

    def _is_supported(self) -> bool:
        return self.filename.lower().endswith(self.SUPPORTED_EXT)

    def process_document(self) -> List[Document]:
        if not self._is_supported():
            raise ValueError(f"Unsupported file type for {self.filename}")

        # load the file into langchain Documents
        if self.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
        elif self.filename.lower().endswith(".txt"):
            # TextLoader requires encoding sometimes; choose appropriate encoding
            loader = TextLoader(self.file_path, encoding="utf-8")
            documents = loader.load()
        else:
            # defensive; should never hit because of _is_supported
            raise ValueError("Unsupported file type")

        # split the text into chunks
        logging.info(f"Splitting document into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        logging.info(f"Document contains {len(documents)} pages/sections before splitting.")    
        chunks = text_splitter.split_documents(documents)
        return chunks