from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_directory = "./chroma_db"

        # load or create DB
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def store_documents(self, docs):

        # check existing vectors
        existing_count = self.vector_db._collection.count()

        print("Existing vectors:", existing_count)

        if existing_count > 0:
            print("Documents already indexed. Skipping insertion.")
            return self.vector_db

        print("Indexing documents...")

        # add documents only if DB empty
        self.vector_db.add_documents(docs)

        print("New vector count:", self.vector_db._collection.count())

        return self.vector_db


    def load_db(self):

        return self.vector_db