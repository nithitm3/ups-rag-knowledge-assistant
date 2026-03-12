from document_ingestion import IngestionPipeline
from rag_pipeline import RAGPipeline
from vector_store import VectorStore

pipeline = IngestionPipeline()
vector_store = VectorStore()

docs = pipeline.ingest("uploads/documents.pdf")

print(f"Number of documents to store: {len(docs)}")

vector_db = vector_store.store_documents(docs)

print(vector_db._collection.count())

rag = RAGPipeline(vector_db, docs)

query = "What is UPS sustainability strategy?"

answer, metadata = rag.generate_answer(query)

print("\nAnswer:", answer)

print("\nSources:")
for m in metadata:
        print(m)