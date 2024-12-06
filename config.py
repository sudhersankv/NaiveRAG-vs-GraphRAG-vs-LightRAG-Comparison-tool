import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API key

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"  # Change this to match the exact model name you pulled

# Default settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

# Neo4j settings
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphrag123"  # This matches the password we set in the Docker command

