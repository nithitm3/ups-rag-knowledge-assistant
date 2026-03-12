import chainlit as cl
from document_ingestion import IngestionPipeline
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Load RAG system
pipeline = IngestionPipeline()
vector_store = VectorStore()

docs = pipeline.ingest("uploads/documents.pdf")
vector_db = vector_store.store_documents(docs)

rag = RAGPipeline(vector_db, docs)


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello 👋 I am your UPS Knowledge Assistant. Ask me anything about the UPS sustainability report."
    ).send()


@cl.on_message
async def main(message: cl.Message):

    query = message.content

    answer, metadata = rag.generate_answer(query)

    # If fallback response → don't show sources
    if "I’m unable to find relevant information for this question in the provided document." in answer:

        await cl.Message(content=answer).send()

    else:

        sources = "\n".join(
            [
                f"Source: {m.get('source','')} | Section: {m.get('section','')} | Chunk: {m.get('chunk_id','')}"
                for m in metadata
            ]
        )

        response = f"""
{answer}

📚 **Sources**
{sources}
"""

        await cl.Message(content=response).send()