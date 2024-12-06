import os
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

class LightRAGWrapper:
    def __init__(self, working_dir="./lightrag_cache"):
        """Initialize RAG with Ollama configuration"""
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Initialize Ollama
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL
        )
        
        # Initialize Ollama Embeddings
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vector store
        self.vectorstore = None
        
        # Track document statistics
        self.entity_count = 0
        self.relation_count = 0
        
    def process_document(self, uploaded_file) -> str:
        """Process document and create vector store"""
        try:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                text_content = uploaded_file.read().decode('utf-8')
            elif file_extension == 'pdf':
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            else:
                return f"Unsupported file extension: {file_extension}"
            
            # Split text into chunks
            texts = self.text_splitter.split_text(text_content)
            
            # Create vector store
            self.vectorstore = FAISS.from_texts(
                texts,
                self.embeddings
            )
            
            # Update statistics
            self.entity_count = len(texts)
            self.relation_count = len(texts) - 1  # Simple relationship count
            
            return self.verify_graph()
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            print(error_msg)
            return error_msg
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the knowledge base using Ollama"""
        try:
            if not self.vectorstore:
                return "Please process a document first!"
            
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
            error_msg = f"Error querying knowledge base: {str(e)}"
            print(error_msg)
            return error_msg
    
    def verify_graph(self) -> str:
        """Return statistics about the processed document"""
        try:
            if not self.vectorstore:
                return "No document has been processed yet."
            
            return f"Document Status:\n- Chunks: {self.entity_count}\n- Connections: {self.relation_count}"
            
        except Exception as e:
            error_msg = f"Error verifying document status: {str(e)}"
            print(error_msg)
            return error_msg 