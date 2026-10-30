from fastapi import Request

from app.config import settings
from app.services.llm_service import LLMService
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstores.chroma_store import ChromaVectorStore
from app.rag.splitters.recursive import RecursiveCharacterTextSplitter
from app.rag.embeddings.gemini import GeminiEmbeddings

IN_MEMORY_DB = {}

def get_db():
    return IN_MEMORY_DB


def get_llm_service(request: Request)->LLMService:
    return LLMService(client=request.app.state.gemini)


def build_rag_pipeline()->RAGPipeline:
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    embeddings = GeminiEmbeddings(settings.GEMINI_API_KEY, output_dimensionality=settings.OUTPUT_DIM, batch_size=settings.BATCH_SIZE)
    store = ChromaVectorStore(embeddings, persist_directory="data/chroma")
    return RAGPipeline(splitter=splitter, vectorstore=store)


def get_rag_pipeline(request: Request)->RAGPipeline:
    return request.app.state.rag