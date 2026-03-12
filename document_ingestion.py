import os
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class IngestionPipeline:

    def __init__(self):

        self.converter = DocumentConverter()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # create markdown folder
        os.makedirs("markdown", exist_ok=True)

    # ---------------------------------------------------
    # PDF → Markdown
    # ---------------------------------------------------

    def pdf_to_markdown(self, file_path):

        try:
            result = self.converter.convert(file_path)

            markdown_text = result.document.export_to_markdown()

            # save markdown for debugging
            filename = os.path.basename(file_path).replace(".pdf", ".md")

            with open(f"markdown/{filename}", "w", encoding="utf-8") as f:
                f.write(markdown_text)

            return markdown_text, filename

        except Exception as e:
            raise Exception(f"PDF conversion failed: {str(e)}")

    # ---------------------------------------------------
    # Markdown → Header chunks → Semantic chunks
    # ---------------------------------------------------

    def chunk_markdown(self, markdown_text, filename):

        headers = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers
        )

        md_chunks = markdown_splitter.split_text(markdown_text)

        final_docs = []

        for i, doc in enumerate(md_chunks):

            section = (
                doc.metadata.get("Header3")
                or doc.metadata.get("Header2")
                or doc.metadata.get("Header1")
                or "Unknown"
            )

            final_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": filename,
                        "section": section,
                        "chunk_id": i
                    }
                )
            )

        return final_docs

    # ---------------------------------------------------
    # Full ingestion pipeline
    # ---------------------------------------------------

    def ingest(self, file_path):

        markdown_text, filename = self.pdf_to_markdown(file_path)

        docs = self.chunk_markdown(markdown_text, filename)

        return docs