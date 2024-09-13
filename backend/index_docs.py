import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging

logging.basicConfig(level=logging.INFO)
COLLECTION_NAME = "document_index"
#for multilingual embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

#to load n split to chunks

def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return [text.page_content for text in texts]

#to create collection
def create_collection_if_not_exists(collection_name, vector_size):
    try:
        qdrant_client.get_collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.info(f"Creating collection '{collection_name}'.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

#embedding
def index_documents(file_path):
    texts = load_and_split_document(file_path)
    create_collection_if_not_exists(COLLECTION_NAME, model.get_sentence_embedding_dimension())
    
    for i, text in enumerate(texts):
        vector = model.encode([text])[0].tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=i, vector=vector, payload={"text": text})]
        )
        logging.info(f"Indexed chunk {i}: {text[:30]}...")

if __name__ == "__main__":
    english_doc_path = "docs/Executive Regulation Law No 6-2016 - English[1].pdf"
    arabic_doc_path = "docs/Executive Regulation Law No 6-2016[1].pdf"
    
    index_documents(english_doc_path)
    index_documents(arabic_doc_path)