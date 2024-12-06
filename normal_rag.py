from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
import PyPDF2
import requests

class NormalRAG:
    def __init__(self):
        try:
            # Test Ollama connection
            self.test_ollama_connection()
            
            # Initialize components
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # Initialize Ollama embeddings
            self.embeddings = OllamaEmbeddings(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            
            # Initialize Ollama LLM
            self.llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.7
            )
            
            self.vectorstore = None
            
        except Exception as e:
            raise Exception(f"Error initializing RAG: {str(e)}")
    
    def test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                raise Exception("Cannot connect to Ollama server")
            
            models = response.json().get("models", [])
            if not any(OLLAMA_MODEL in model.get("name", "") for model in models):
                raise Exception(f"Model {OLLAMA_MODEL} not found. Please run: ollama pull {OLLAMA_MODEL}")
                
        except requests.exceptions.RequestException:
            raise Exception("Ollama server is not running")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        
    def extract_text_from_csv(self, csv_file):
        """Extract text from CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            # Combine all columns into a single text
            text = "\n".join([
                ", ".join(map(str, row)) 
                for row in df.values
            ])
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from CSV: {str(e)}")
    
    def process_document(self, file):
        """Process the input document and create vector store"""
        try:
            # Get file type from extension
            file_type = file.name.split('.')[-1].lower()
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = self.extract_text_from_pdf(file)
            elif file_type == 'csv':
                text = self.extract_text_from_csv(file)
            else:
                return f"Unsupported file type: {file_type}"
            
            if not text.strip():
                return "Error: Extracted text is empty. Please check if the file is readable."
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                return "Error: No text chunks were created. Please check the document content."
            
            # Create vector store
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
            return "Document processed successfully!"
            
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def query(self, question):
        """Query the processed document"""
        if self.vectorstore is None:
            return "Please process a document first!"
        
        try:
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            # Get response
            response = qa_chain.run(question)
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}" 