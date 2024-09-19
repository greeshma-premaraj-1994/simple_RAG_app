from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GPTNeoForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from operator import itemgetter
from pydantic import BaseModel
from langdetect import detect
import logging
# import openai
import os
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load the OpenAI API key from environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

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
english_vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="english_documents_collection",
    url="http://localhost:6333"
)
arabic_vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="arabic_documents_collection",
    url="http://localhost:6333"
)
# Set up memory for conversational retrieval
memory = ConversationBufferMemory()

# Load GPT-Neo model and tokenizer for text generation
gpt_neo_model_name = "EleutherAI/gpt-neo-1.3B"
gpt_neo_model = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name)
gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_model_name)

# Create a Hugging Face pipeline for text generation
llm_pipeline = pipeline("text-generation", model=gpt_neo_model, tokenizer=gpt_neo_tokenizer,max_length=256,max_new_tokens=200,truncation=True, num_return_sequences=1)
llm = HuggingFacePipeline(
    pipeline=llm_pipeline
)
# # Define a simple function to interact with the model
# def generate_text(prompt, max_length=200):
#     result = llm_pipeline(prompt, max_length=max_length, num_return_sequences=1)
#     return result[0]["generated_text"]

# Define the PromptTemplate for combining document context and question
prompt_template = """You are a helpful assistant. Use the following documents to answer the question:

{context}

Question: {question}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain_1 = (
    {
        "context": itemgetter("question") | english_vector_store.as_retriever(),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
# rag1=rag_chain_1.invoke({"question": "what is article 1?"})

# logging.info(f"invoked....{rag1}")


# Chain
llm_chain = prompt | llm | StrOutputParser()
rag_chain_2 = MultiQueryRetriever(
    retriever=arabic_vector_store.as_retriever(), llm_chain=llm_chain
)
# rag2 = rag_chain_2.invoke("عمله الأصلية.")
# print("len....",len(rag2))



def route(info):
    language = detect(info["question"])
    if language == 'en':
        return rag_chain_1
    elif language == 'ar':
        return rag_chain_2
    else:
        return "I am sorry, I am not allowed to answer in this language."


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
            full_chain = {"question": query} | RunnableLambda(route)
            response = full_chain.invoke({"question": query})
            print("response...",response)
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