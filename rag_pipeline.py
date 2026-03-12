from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


class RAGPipeline:

    def __init__(self, vector_db, docs):

        load_dotenv()

        # Embedding model (used for reranking)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector retriever
        self.vector_retriever = vector_db.as_retriever(
            search_kwargs={"k": 8}
        )

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 8

        # LLM
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        # Croos-encoder for reranking
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")

    # -----------------------------
    # Hybrid Retrieval
    # -----------------------------
    def retrieve_documents(self, query):

        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        seen = set()
        combined_docs = []

        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_docs.append(doc)

        return combined_docs

    # -----------------------------
    # Cosine Similarity Reranking
    # -----------------------------
    def rerank_documents(self, query, docs, top_k=8):

        query_embedding = self.embeddings.embed_query(query)

        doc_embeddings = [
            self.embeddings.embed_query(doc.page_content)
            for doc in docs
        ]

        scores = cosine_similarity([query_embedding], doc_embeddings)[0]

        ranked_docs = list(zip(docs, scores))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:top_k]]
    
    # -----------------------------
    # Cross-Encoder Reranking
    # -----------------------------
    def cross_encoder_rerank(self, query, docs, top_k=3):

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.reranker.predict(pairs)

        ranked_docs = list(zip(docs, scores))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, _ in ranked_docs[:top_k]]

        return reranked_docs

    # -----------------------------
    # Build Context
    # -----------------------------
    def retrieve_context(self, query):

        # Step 1: Hybrid retrieval
        docs = self.retrieve_documents(query)

        # Step 2: Cosine similarity reranking
        cosine_reranked = self.rerank_documents(query, docs, top_k=8)

        # Step 3: Cross-encoder reranking
        final_docs = self.cross_encoder_rerank(query, cosine_reranked, top_k=3)

        context = "\n\n".join([doc.page_content for doc in final_docs])
        metadata = [doc.metadata for doc in final_docs]

        return context, metadata

    # -----------------------------
    # Generate Answer
    # -----------------------------
    def generate_answer(self, query):

        context, metadata = self.retrieve_context(query)

        if not context.strip():
            return "I’m unable to find relevant information for this question in the provided document.", metadata

        template = """
        You are a helpful AI assistant.

        You must answer the question ONLY using the provided context.

        Rules:
        1. Do not use external knowledge.
        2. If the answer cannot be found in the context, respond with:
        "I’m unable to find relevant information for this question in the provided document."
        3. Do not guess.
        4. Keep the answer concise.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "query": query
        })

        return response, metadata