"""
Data processing module for PDF ingestion and vector store creation
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles PDF processing and vector store creation"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_pdfs(self, pdf_directory: Path) -> List[Document]:
        """Load and process all PDFs in the directory"""
        documents = []
        pdf_files = list(pdf_directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_file': pdf_path.name,
                        'file_path': str(pdf_path),
                        'source_type': 'pdf'
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
                
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content)
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> Any:
        """Create and save vector store"""
        logger.info("Creating vector store...")
        
        if config.vector_store_type.lower() == "faiss":
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(str(config.vector_store_dir))
            
        elif config.vector_store_type.lower() == "chroma":
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(config.vector_store_dir)
            )
            vector_store.persist()
            
        else:
            raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
        
        logger.info(f"Vector store created with {len(chunks)} documents")
        return vector_store
    
    def load_vector_store(self) -> Any:
        """Load existing vector store"""
        if not config.vector_store_dir.exists():
            raise FileNotFoundError("Vector store not found. Please run data processing first.")
        
        if config.vector_store_type.lower() == "faiss":
            vector_store = FAISS.load_local(
                str(config.vector_store_dir), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif config.vector_store_type.lower() == "chroma":
            vector_store = Chroma(
                persist_directory=str(config.vector_store_dir),
                embedding_function=self.embeddings
            )
        else:
            raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
        
        return vector_store
    
    def process_all(self, pdf_directory: Path = None) -> Any:
        """Complete processing pipeline"""
        if pdf_directory is None:
            pdf_directory = config.pdf_dir
            
        # Load PDFs
        documents = self.load_pdfs(pdf_directory)
        if not documents:
            raise ValueError("No documents loaded")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Save processing metadata
        metadata = {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "vector_store_type": config.vector_store_type
        }
        
        metadata_path = config.data_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Data processing completed successfully")
        return vector_store

def main():
    """CLI for data processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs and create vector store")
    parser.add_argument("--pdf-dir", type=Path, default=config.pdf_dir,
                       help="Directory containing PDF files")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing even if vector store exists")
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    # Check if vector store already exists
    if config.vector_store_dir.exists() and not args.force:
        logger.info("Vector store already exists. Use --force to reprocess.")
        return
    
    try:
        vector_store = processor.process_all(args.pdf_dir)
        logger.info("✅ Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        raise

if __name__ == "__main__":
    main()