import streamlit as st
from normal_rag import NormalRAG
from custom_graph_rag import GraphRAG
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from light_rag import LightRAGWrapper

def set_custom_style():
    st.markdown("""
        <style>
        .main-header {
            color: #1565C0;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 30px;
            text-align: center;
        }
        .method-title {
            color: #0D47A1;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .query-section {
            background-color: #E0E0E0;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .response-section {
            background-color: #C8E6C9;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_rag(method):
    if method == "Normal RAG":
        return NormalRAG()
    else:
        return GraphRAG(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

def main():
    # Initialize session state
    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False

    set_custom_style()
    st.markdown('<p class="main-header">📚 Document Q&A with RAG</p>', unsafe_allow_html=True)

    # Method selection and information
    method = st.selectbox(
        "Choose which RAG method to use:",
        ["Normal RAG", "GraphRAG", "LightRAG"],
        help="Select the RAG method you want to test",
        key="rag_type"
    )

    # Display method-specific information
    if method == "Normal RAG":
        st.info("""
        📚 **Normal RAG Process:**
        - Splits document into chunks
        - Creates embeddings for each chunk
        - Uses vector similarity for retrieval
        - Best for simple Q&A tasks
        """)
    elif method == "GraphRAG":
        st.info("""
        🕸️ **GraphRAG Process:**
        - Maintains relationships between concepts
        - Creates knowledge graph structure
        - Uses graph traversal for context
        - Better for complex, relationship-based queries
        """)
    else:
        st.info("""
        🌐 **LightRAG Process:**
        - Uses LightRAG's built-in features
        - Efficient document processing
        - Built-in caching
        - Advanced relationship extraction
        """)

    # Initialize or update RAG instance
    if st.session_state.rag_instance is None or st.session_state.rag_type != getattr(st.session_state, 'last_rag_type', None):
        st.session_state.rag_instance = initialize_rag(method)
        st.session_state.last_rag_type = method

    # Clear Graph button for GraphRAG
    if isinstance(st.session_state.rag_instance, GraphRAG):
        if st.button("Clear Graph", type="secondary"):
            with st.spinner('Clearing knowledge graph...'):
                st.session_state.rag_instance.clear_graph()
                st.session_state.document_processed = False
                st.rerun()

    # File upload and processing
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'csv'])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner('Processing document...'):
                if method == "LightRAG":
                    rag = LightRAGWrapper(working_dir="./lightrag_cache")
                    result = rag.process_document(uploaded_file)
                    st.write(result)
                    
                    if st.button("Query Knowledge Graph"):
                        question = st.text_input("Enter your question:")
                        if question:
                            modes = ["hybrid", "local", "global", "naive"]
                            selected_mode = st.selectbox("Select query mode:", modes)
                            response = rag.query(question, mode=selected_mode)
                            st.write("Response:", response)
                else:
                    result = st.session_state.rag_instance.process_document(uploaded_file)
                    if "successfully" in result.lower():
                        st.success(result)
                        st.session_state.document_processed = True
                    else:
                        st.error(result)

    # Query section
    st.markdown('<p class="method-title">Enter Your Query</p>', unsafe_allow_html=True)
    query = st.text_area("Type your question here:", height=100)

    # Submit query
    if st.button("Submit Query", type="primary"):
        if not uploaded_file:
            st.warning("⚠️ Please upload a document first!")
        elif not st.session_state.document_processed:
            st.warning("⚠️ Please process the document first!")
        elif not query:
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner('Generating response...'):
                response = st.session_state.rag_instance.query(query)
                st.write(response)

if __name__ == "__main__":
    main() 