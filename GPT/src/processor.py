from typing import List, Dict
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.vector_store = None

    def load_documents(self, json_file: str) -> List[Document]:
        """Load documents from JSON file and convert to Langchain Document format."""
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_docs = json.load(f)

        documents = []
        for doc in raw_docs:
            text = f"Title: {doc['title']}\n\nContent: {doc['content']}"
            metadata = {'source': doc['url'], 'title': doc['title']}
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def process_documents(self, documents: List[Document]) -> None:
        """Split documents and create vector store."""
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./data/vector_store"
        )

    def search_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents based on query."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process documents first.")

        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score)
            })
        
        return formatted_results

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    documents = processor.load_documents('data/scraped_docs.json')
    processor.process_documents(documents)
    
    # Example search
    results = processor.search_documents("How to create a loyalty program?")
