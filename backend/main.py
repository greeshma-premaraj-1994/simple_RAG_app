from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.chains import RetrievalQA, RAGChain, MultiQueryRetriever, ChainRouter
# from langchain.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import LangChainEmbeddings

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging
import openai
import os

# Load the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client = QdrantClient(host="localhost", port=6333)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(embedding_model_name)
# def embed_texts(texts):
#     return model.encode(texts, convert_to_tensor=True).tolist()
# embeddings = embed_texts('')
vector_store = QdrantVectorStore.from_existing_collection(
    embeddings=embeddings,
    collection_name="document_index",
    url="http://localhost:6333",
)

memory = ConversationBufferMemory()

retrieval_chain = ConversationalRetrievalChain(
    retriever=vector_store.as_retriever(),
    memory=memory,
    return_source_documents=True
)

#websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established")
    try:
        while True:
            query = await websocket.receive_text()
            logging.info(f"Received query: {query}")


            response = retrieval_chain.run(query)

            await websocket.send_text(response['output_text'])
    except WebSocketDisconnect:
        logging.info("WebSocket connection closed")
    except Exception as e:
        logging.error(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)