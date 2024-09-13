from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.chains import ConversationalRetrievalChain, ConversationBufferMemory
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging

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

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(embedding_model_name)

vector_store = Qdrant(
    qdrant_client=qdrant_client, 
    collection_name="document_index", 
    embedding_function=model.encode  
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