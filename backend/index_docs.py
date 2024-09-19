import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO)

# Define constants 
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
ARABIC_COLLECTION_NAME = os.getenv("ARABIC_COLLECTION_NAME")
ENGLISH_COLLECTION_NAME = os.getenv("ENGLISH_COLLECTION_NAME")
ARABIC_DOC_PATH = os.getenv("ARABIC_DOC_PATH")
ENGLISH_DOC_PATH = os.getenv("ENGLISH_DOC_PATH")


# Initialize Qdrant client for interacting with the vector store
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

# Initialize SentenceTransformer model for generating embeddings
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


#  -------------------------Document Augmentation and Indexing:-------------------------------------------------

# Function to load and split a PDF document into chunks
def load_and_split_document(file_path):
    # Load the document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the loaded document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Return the chunked text content
    return [text.page_content for text in texts]

# Function to create a Qdrant collection if it does not already exist
def create_collection_if_not_exists(collection_name, vector_size):
    try:
        # Check if the collection already exists
        qdrant_client.get_collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # If not, create a new collection
        logging.info(f"Creating collection '{collection_name}'.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            # vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

# Function to index documents by converting them into vectors and storing them in Qdrant
def index_documents(file_path,COLLECTION_NAME):
    # Load and split the document into text chunks
    texts = load_and_split_document(file_path)
    print("model.get_sentence_embedding_dimension()",model.get_sentence_embedding_dimension())
    # Create the collection in Qdrant if it doesn't exist
    create_collection_if_not_exists(COLLECTION_NAME, model.get_sentence_embedding_dimension())
    
    
    # Iterate over each text chunk and index it
    for i, text in enumerate(texts):
        # Generate embedding vector for the text chunk
        vector = model.encode([text])[0].tolist()
        
        # Upsert the vector into the Qdrant collection
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=i, vector=vector, payload={"text": text})]
        )
        logging.info(f"Indexed chunk {i}: {text[:30]}...")  # Log a brief snippet of the indexed text

# Main execution block
if __name__ == "__main__":
    # Define file paths for the documents to be indexed
    english_doc_path = ENGLISH_DOC_PATH
    arabic_doc_path = ARABIC_DOC_PATH
    
    # Index both the English and Arabic documents
    index_documents(english_doc_path,ENGLISH_COLLECTION_NAME)
    index_documents(arabic_doc_path,ARABIC_COLLECTION_NAME)