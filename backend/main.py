from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GPTNeoForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from qdrant_client import QdrantClient
from pydantic import BaseModel
import logging
# import openai
import os
from langchain_huggingface import HuggingFaceEmbeddings

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
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="document_index",
    url="http://localhost:6333"
)

# Set up memory for conversational retrieval
memory = ConversationBufferMemory()

# Load GPT-Neo model and tokenizer for text generation
gpt_neo_model_name = "EleutherAI/gpt-neo-1.3B"
gpt_neo_model = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name)
gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_model_name)

# Create a Hugging Face pipeline for text generation
llm_pipeline = pipeline("text-generation", model=gpt_neo_model, tokenizer=gpt_neo_tokenizer)

# Define a simple function to interact with the model
def generate_text(prompt, max_length=200):
    result = llm_pipeline(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"]

# Define the PromptTemplate for combining document context and question
prompt_template = """You are a helpful assistant. Use the following documents to answer the question:

{context}

Question: {question}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# The combine_docs_chain will be the text generation logic using GPT-Neo
combine_docs_chain = LLMChain(llm=lambda inputs: {"generated_text": generate_text(inputs['context'] + "\nQuestion: " + inputs['question'])}, prompt=prompt)

# Load T5 model for question generation
t5_model_name = "t5-small"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Define function to generate questions using T5
def generate_question(context):
    input_text = f"generate question: {context}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5_model.generate(input_ids, max_length=64, num_return_sequences=1)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Create the Conversational Retrieval Chain with question generator
retrieval_chain = ConversationalRetrievalChain(
    retriever=vector_store.as_retriever(),
    combine_docs_chain=combine_docs_chain,
    question_generator=LLMChain(
        llm=lambda inputs: {"generated_question": generate_question(inputs['context'])},
        prompt=PromptTemplate(template="{context}", input_variables=["context"])
    ),
    memory=memory,
    return_source_documents=True
)
# Create ConversationalRetrievalChain with retriever and memory
# retrieval_chain = ConversationalRetrievalChain(
#     retriever=vector_store.as_retriever(),
#     memory=memory,
#     return_source_documents=True
# )

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