from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging
import openai
import os

# Load the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS Middleware configuration to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize SentenceTransformer for embeddings
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(embedding_model_name)

# Define embedding function
def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=True).tolist()

# Create Qdrant vector store from existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embeddings=embed_texts,
    collection_name="document_index",
    url="http://localhost:6333",
)

# Set up memory for conversational retrieval
memory = ConversationBufferMemory()

# Create ConversationalRetrievalChain with retriever and memory
retrieval_chain = ConversationalRetrievalChain(
    retriever=vector_store.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# WebSocket endpoint for real-time query processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established")
    try:
        while True:
            query = await websocket.receive_text()
            logging.info(f"Received query: {query}")

            # Retrieve response from the retrieval chain
            response = retrieval_chain.run(query)

            # Send response back to the WebSocket client
            await websocket.send_text(response['output_text'])
    except WebSocketDisconnect:
        logging.info("WebSocket connection closed")
    except Exception as e:
        logging.error(f"Error: {e}")
        await websocket.close()

# Entry point for running the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)