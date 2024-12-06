
# **GraphRAG: Advanced Document Question-Answering System**

GraphRAG is a cutting-edge document question-answering system that supports multiple Retrieval-Augmented Generation (RAG) approaches, combining traditional vector-based retrieval with graph-based knowledge representation for enhanced query processing.

---

## **🌟 Key Features**

- **Multi-RAG Implementations**:
  - **Normal RAG**: Vector-based retrieval for traditional Q&A.
  - **GraphRAG**: Graph-based knowledge representation for relationship-oriented queries.
  - **LightRAG**: Lightweight, efficient, local processing.

- **Document Processing**:
  - Supports **PDF** and **CSV** files.
  - Automatic entity and relationship extraction.

- **Local LLM Integration**:
  - Powered by **Ollama** for local LLM inference.
  - Compatible with various **Llama models**.

- **Interactive UI**:
  - Built with **Streamlit**.
  - Real-time query processing and graph visualization.
  - RAG-specific information feedback.

---

## **🛠️ Technical Architecture**

### **Normal RAG**
- Uses **LangChain** for document processing.
- **FAISS** for vector storage.
- **Ollama** for embeddings and LLM.
- Recursive character text splitting for efficient chunking.

### **GraphRAG**
- **Neo4j** for knowledge graph storage.
- **spaCy** for Named Entity Recognition (NER) and relationship extraction.
- Flexible handling of graph data with support for both entities and relationships.

### **LightRAG**
- Optimized for local and lightweight usage.
- Streamlined entity and relationship extraction for resource-constrained environments.

---

## **🚀 Getting Started**

### **1. Prerequisites**
Ensure the following tools and libraries are installed:
- Python 3.10 or later
- Neo4j
- Streamlit
- Ollama

### **2. Installation**

#### Clone the Repository:
```bash
git clone https://github.com/your-username/GraphRAG.git
cd GraphRAG
```

#### Install Python Dependencies:
```bash
pip install -r requirements.txt
```

#### Install spaCy Model:
```bash
python -m spacy download en_core_web_sm
```

#### Pull Llama Model for Ollama:
```bash
ollama pull llama3.1:8b
```

#### Set Up Neo4j:
```bash
docker run \
--name neo4j \
-p 7474:7474 -p 7687:7687 \
-d \
-e NEO4J_AUTH=neo4j/graphrag123 \
neo4j:latest
```

---

## **🌐 Running the Application**

Start the Streamlit app:
```bash
streamlit run app.py
```

1. Select a RAG Implementation:
   - **Normal RAG** for general Q&A.
   - **GraphRAG** for relationship-based queries.
   - **LightRAG** for efficient local processing.

2. Upload your document (PDF or CSV):
   - Extracts text, identifies entities, and maps relationships.

3. Enter a natural language query:
   - Get context-aware responses.
   - Visualize relationship graphs (GraphRAG).

---

## **🎨 Visualization**

### **Entity Colors**
| Entity Type     | Color          | Hex Code  |
|------------------|----------------|-----------|
| Person           | Light Pink     | `#FFB6C1` |
| Organization     | Pale Green     | `#98FB98` |
| Location         | Sky Blue       | `#87CEEB` |
| Date             | Plum           | `#DDA0DD` |
| Money            | Khaki          | `#F0E68C` |
| Percentage       | Lavender       | `#E6E6FA` |
| Product          | Gold           | `#FFD700` |
| Event            | Light Salmon   | `#FFA07A` |
| Facility         | Light Sea Green| `#20B2AA` |

### **Relationship Colors**
| Relationship Type    | Color         | Hex Code  |
|-----------------------|---------------|-----------|
| Works For            | Orange Red    | `#FF4500` |
| Partners With        | Lime Green    | `#32CD32` |
| Interacts With       | Royal Blue    | `#4169E1` |
| Located In           | Medium Purple | `#9370DB` |
| Happened On          | Hot Pink      | `#FF69B4` |
| Valued At            | Dark Turquoise| `#00CED1` |
| Measured As          | Orange        | `#FFA500` |

---

## **📂 File Structure**

```plaintext
GraphRAG/
├── app.py                   # Main Streamlit application
├── normal_rag.py            # Normal RAG implementation
├── custom_graph_rag.py      # GraphRAG implementation
├── light_rag.py             # LightRAG implementation
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## **🤝 Contributing**

Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature-name"`.
4. Push to your fork: `git push origin feature-name`.
5. Open a pull request.

---

## **📜 License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **📧 Contact**

For queries, reach out to:
- **Email**: your-email@example.com
- **GitHub**: [sudhersankv](https://github.com/sudhersankv)

--- 
