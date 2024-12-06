from typing import List, Dict
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
import spacy
import PyPDF2
import io
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_MODEL, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP

class GraphRAG:
    def __init__(self, uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD):
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Initialize NLP
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Initialize LLM components
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        self.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7
        )

    def process_document(self, uploaded_file) -> str:
        """Process uploaded document and create knowledge graph"""
        try:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                print(f"Extracted text: {text[:200]}...")  # Debug print
                
                # Process text directly instead of chunks
                self.create_knowledge_graph(text)
                    
            elif file_extension == 'csv':
                # Reset file pointer if needed
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                
                success = self.process_csv_to_graph(uploaded_file)
                if not success:
                    return "Error processing CSV file"
            else:
                return f"Unsupported file extension: {file_extension}"

            # Verify graph creation with more details
            verification = self.verify_graph()
            if "Nodes: 0" in verification:
                return "No entities or relationships were extracted from the document. Please check if the document contains relevant crypto-related content."
            return f"Document processed successfully. {verification}"

        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            print(error_msg)
            return error_msg

    

    def create_knowledge_graph(self, text: str):
        """Create knowledge graph with colored entities and relationships"""
        try:
            # Define color schemes
            entity_colors = {
                "PERSON": "#FFB6C1",     # Light pink
                "ORG": "#98FB98",        # Pale green
                "GPE": "#87CEEB",        # Sky blue
                "DATE": "#DDA0DD",       # Plum
                "MONEY": "#F0E68C",      # Khaki
                "PERCENT": "#E6E6FA",    # Lavender
                "PRODUCT": "#FFD700",    # Gold
                "EVENT": "#FFA07A",      # Light salmon
                "FAC": "#20B2AA",        # Light sea green
                "DEFAULT": "#D3D3D3"     # Light gray
            }
            
            relationship_colors = {
                "WORKS_FOR": "#FF4500",      # Orange red
                "PARTNERS_WITH": "#32CD32",   # Lime green
                "INTERACTS_WITH": "#4169E1",  # Royal blue
                "LOCATED_IN": "#9370DB",      # Medium purple
                "HAPPENED_ON": "#FF69B4",     # Hot pink
                "VALUED_AT": "#00CED1",       # Dark turquoise
                "MEASURED_AS": "#FFA500",     # Orange
                "MENTIONS": "#A9A9A9"         # Dark gray
            }
            
            extracted_info = self.extract_entities_and_relations(text)
            
            with self.driver.session() as session:
                # First, set style constraints
                session.run("""
                    CALL db.labels() YIELD label
                    RETURN count(label)
                """)
                
                # Create colored entities
                for entity in extracted_info["entities"]:
                    color = entity_colors.get(entity["label"], entity_colors["DEFAULT"])
                    session.run("""
                        MERGE (e:Entity {text: $text})
                        SET e.label = $label,
                            e.color = $color,
                            e.caption = $text,
                            e.size = 5.0
                        """, 
                        {
                            "text": entity["text"],
                            "label": entity["label"],
                            "color": color
                        }
                    )
                
                # Create colored relationships
                for relation in extracted_info["relations"]:
                    # Get entity labels
                    result = session.run("""
                        MATCH (s:Entity {text: $subject})
                        MATCH (o:Entity {text: $object})
                        RETURN s.label as subject_label, o.label as object_label
                        """,
                        {
                            "subject": relation["subject"],
                            "object": relation["object"]
                        }
                    )
                    
                    labels = result.single()
                    if labels:
                        subj_label = labels["subject_label"]
                        obj_label = labels["object_label"]
                        
                        # Determine relationship type
                        if subj_label == "PERSON" and obj_label == "ORG":
                            rel_type = "WORKS_FOR"
                        elif subj_label == "ORG" and obj_label == "ORG":
                            rel_type = "PARTNERS_WITH"
                        elif subj_label == "PERSON" and obj_label == "PERSON":
                            rel_type = "INTERACTS_WITH"
                        elif "GPE" in [subj_label, obj_label]:
                            rel_type = "LOCATED_IN"
                        elif "DATE" in [subj_label, obj_label]:
                            rel_type = "HAPPENED_ON"
                        elif "MONEY" in [subj_label, obj_label]:
                            rel_type = "VALUED_AT"
                        elif "PERCENT" in [subj_label, obj_label]:
                            rel_type = "MEASURED_AS"
                        else:
                            rel_type = "MENTIONS"
                        
                        # Get color for relationship
                        rel_color = relationship_colors[rel_type]
                        
                        # Create the colored relationship
                        session.run(f"""
                            MATCH (s:Entity {{text: $subject}})
                            MATCH (o:Entity {{text: $object}})
                            MERGE (s)-[r:`{rel_type}`]->(o)
                            SET r.predicate = $predicate,
                                r.type = $rel_type,
                                r.color = $color,
                                r.thickness = 2.0,
                                r.caption = $rel_type
                            """,
                            {
                                "subject": relation["subject"],
                                "object": relation["object"],
                                "predicate": relation["predicate"],
                                "rel_type": rel_type,
                                "color": rel_color
                            }
                        )

            # Print styling instructions
            print("\nTo view colors in Neo4j Browser:")
            print("1. Click the database icon in the left sidebar")
            print("2. Click 'Node Labels' and 'Relationship Types'")
            print("3. Click the paint brush icon")
            print("4. Select 'Node Color' and choose 'color' property")
            print("5. Select 'Relationship Color' and choose 'color' property")

        except Exception as e:
            print(f"Error in create_knowledge_graph: {str(e)}")

    def query_knowledge_graph(self, query: str) -> str:
        """Query the knowledge graph using natural language"""
        try:
            with self.driver.session() as session:
                # Get all entities and relationships with dynamic types
                result = session.run("""
                    MATCH (n:Entity)-[r]->(m:Entity)
                    RETURN DISTINCT n.text as source, type(r) as relationship, 
                           r.predicate as predicate, m.text as target
                    LIMIT 20
                """).data()
                
                # Format the context
                if not result:
                    return "No relevant information found in the knowledge graph."
                
                # Create a context string from the graph data
                context = "Knowledge Graph Information:\n"
                for item in result:
                    pred = item['predicate'] if item['predicate'] else item['relationship'].lower()
                    context += f"- {item['source']} {pred} {item['target']}\n"
                
                # Create a prompt for the LLM
                prompt = f"""Based on this knowledge graph information:

{context}

Please answer this question: {query}

If you can't find relevant information in the knowledge graph, please say so."""
                
                # Get response from LLM
                response = self.llm.invoke(prompt)
                return response

        except Exception as e:
            return f"Error querying knowledge graph: {str(e)}"

    def get_graph_statistics(self) -> str:
        """Get basic statistics about the knowledge graph"""
        try:
            with self.driver.session() as session:
                # Count nodes
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                # Count relationships
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                # Get node labels
                labels = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()["labels"]
                
                return f"""
                Knowledge Graph Statistics:
                - Nodes: {node_count}
                - Relationships: {rel_count}
                - Node Types: {', '.join(labels)}
                """
        except Exception as e:
            return f"Error getting graph statistics: {str(e)}"

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def query(self, question: str) -> str:
        """Interface method to match NormalRAG's interface"""
        try:
            if not question:
                return "Please provide a question!"
            
            # Use the knowledge graph query method
            response = self.query_knowledge_graph(question)
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def verify_graph(self) -> str:
        """Verify graph creation with basic queries"""
        try:
            with self.driver.session() as session:
                # Check nodes
                nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                
                # Check relationships
                rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                # Get sample data
                sample = session.run("""
                    MATCH (n)
                    RETURN n.text as text, n.label as label
                    LIMIT 5
                """).data()
                
                return f"""
                Graph Status:
                - Nodes: {nodes}
                - Relationships: {rels}
                - Sample Nodes: {sample}
                """
        except Exception as e:
            return f"Error verifying graph: {str(e)}"

    def process_csv_to_graph(self, csv_file):
        """Process CSV with improved relationship creation"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            with self.driver.session() as session:
                # Create nodes for each column header
                for col in df.columns:
                    session.run("""
                        MERGE (c:Column {name: $name})
                        """, {"name": col})
                
                # Create nodes and relationships for each row
                for idx, row in df.iterrows():
                    # Create node for each cell value
                    for col in df.columns:
                        value = str(row[col])
                        if pd.notna(value) and value.strip():
                            session.run("""
                                MERGE (v:Value {text: $text})
                                SET v.column = $column
                                WITH v
                                MATCH (c:Column {name: $column})
                                MERGE (v)-[r:BELONGS_TO]->(c)
                                """,
                                {"text": value, "column": col}
                            )
                    
                    # Create relationships between adjacent column values
                    for i in range(len(df.columns) - 1):
                        col1, col2 = df.columns[i], df.columns[i+1]
                        val1, val2 = str(row[col1]), str(row[col2])
                        if pd.notna(val1) and pd.notna(val2) and val1.strip() and val2.strip():
                            session.run("""
                                MATCH (v1:Value {text: $val1, column: $col1})
                                MATCH (v2:Value {text: $val2, column: $col2})
                                MERGE (v1)-[r:RELATES_TO {from_col: $col1, to_col: $col2}]->(v2)
                                """,
                                {
                                    "val1": val1, "col1": col1,
                                    "val2": val2, "col2": col2
                                }
                            )
            
            return True
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return False

    def clear_graph(self) -> str:
        """Clear all nodes and relationships from the graph"""
        try:
            with self.driver.session() as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                return "Knowledge graph cleared successfully!"
        except Exception as e:
            return f"Error clearing graph: {str(e)}"

    def extract_entities_and_relations(self, text: str) -> Dict:
        """Extract entities and relationships using spaCy's NER"""
        doc = self.nlp(text)
        entities = []
        relations = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })
            print(f"Found entity: {ent.text} ({ent.label_})")
        
        # Extract relationships between entities in same sentence
        for sent in doc.sents:
            sent_ents = list(sent.ents)
            
            # Find relationships between entities
            for i, ent1 in enumerate(sent_ents):
                for ent2 in sent_ents[i+1:]:
                    # Find verbs between entities
                    between_tokens = doc[ent1.end:ent2.start]
                    verbs = [token.text for token in between_tokens if token.pos_ == "VERB"]
                    
                    if verbs:
                        # Use the first verb as predicate
                        relations.append({
                            "subject": ent1.text,
                            "predicate": verbs[0],
                            "object": ent2.text
                        })
                    else:
                        # Create basic connection if no verb found
                        relations.append({
                            "subject": ent1.text,
                            "predicate": "connected_to",
                            "object": ent2.text
                        })
        
        print(f"\nExtracted {len(entities)} entities and {len(relations)} relations")
        return {"entities": entities, "relations": relations}