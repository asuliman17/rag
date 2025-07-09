import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama # Keep this import for general Ollama checks, though LangChain handles model interaction
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever # Removed for simplicity
from IPython.display import display, Markdown # For display in environments like Jupyter/Colab
from typing import List, Dict # Import for type hints
import shutil # Added for robust DB deletion

# --- Telemetry suppression (add this at the very top) ---
# Prevents Chroma from sending anonymous usage data
os.environ['CHROMA_ANALYTICS'] = 'False'
# --- End telemetry suppression ---

# --- Added for debugging LangChain (optional, uncomment to enable) ---
# from langchain.globals import set_debug, set_verbose
# set_debug(True)
# set_verbose(True)
# --- End debugging setup ---

class llm:
    def __init__(self):
        self.all_documents = [] # This might not be strictly needed if directly using DB
        self.vector_db = None

    def load_and_split_document(self, file_path: str, doc_title: str) -> list[Document]:
        """
        Loads a PDF document and splits it into smaller chunks.
        Adds relevant metadata to each chunk.
        """
        print(f"Loading document: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            if not pages:
                print(f"Warning: No pages loaded from {file_path}. Check file path and content.")
                return []
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []

        # --- IMPORTANT CHANGE: Reduced chunk_size for better RAG performance ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Max size for each chunk (was 7500)
            chunk_overlap=100, # Overlap between chunks for context
            add_start_index=True # Add character index of the chunk in the original document
        )
        chunks = text_splitter.split_documents(pages)

        # Add custom metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "title": doc_title,
                "author": "company", # Example author
                "date": str(datetime.date.today()), # Current date
                "chunk_number": i + 1 # Assign a unique chunk number
            })
            # Ensure 'source' metadata is present, using filename if not already set by loader
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = os.path.basename(file_path)

        print(f"Split {len(chunks)} chunks from {file_path} with metadata.")
        return chunks

    def store_in_vector_db(self, documents: list[Document], db_path: str):
        """
        Stores a list of Document objects in a Chroma vector database.
        Embeddings are generated using Ollama's 'nomic-embed-text' model.
        """
        os.makedirs(db_path, exist_ok=True) # Ensure the directory exists
        print(f"Storing {len(documents)} documents in vector DB at {db_path}...")

        # Initialize OllamaEmbeddings for the embedding model
        # Ensure 'nomic-embed-text' is pulled and Ollama server is running
        embeddings = OllamaEmbeddings(model='nomic-embed-text')

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_path
        )
        self.vector_db.persist() # Explicitly persist the database to disk
        print(f"Vector DB stored successfully at {db_path}")
        return self.vector_db

    def load_vector_db(self, db_path: str, model_name='nomic-embed-text') -> bool:
        """
        Attempts to load an existing Chroma vector database from the specified path.
        Returns True if successful, False otherwise.
        """
        # Check if the directory exists and contains files (indicating a persisted DB)
        if os.path.exists(db_path) and any(os.path.isfile(os.path.join(db_path, f)) for f in os.listdir(db_path)):
            print(f"Loading existing vector DB from {db_path}...")
            try:
                self.vector_db = Chroma(
                    persist_directory=db_path,
                    embedding_function=OllamaEmbeddings(model=model_name)
                )
                # Perform a small test query to ensure it loaded correctly and is functional
                # This helps catch issues where directory exists but DB is corrupt/empty
                test_query = "test"
                results = self.vector_db.similarity_search_with_score(test_query, k=1)
                print(f"Successfully loaded vector DB from {db_path}. Test query results count: {len(results)}")
                return True
            except Exception as e:
                print(f"Error loading vector DB from {db_path}: {e}")
                print("It might be corrupted or incompatible. Consider deleting it and recreating.")
                return False
        else:
            print(f"Vector DB not found at {db_path} or directory is empty.")
            return False

    def multiquery_retriever(self): # Function name kept for consistency, but now uses simple retriever
        """
        Sets up and executes a simple RAG chain using the loaded vector DB and Ollama.
        """
        if not self.vector_db:
            print("Error: Vector DB not loaded or created. Please ensure it's available before running the retriever.")
            return

        local_model = "llama3" # Ensure 'llama3' model is pulled for Ollama
        llm_model = ChatOllama(model=local_model, temperature=0.1) # Lower temperature for factual answers

        # Define the RAG prompt for the final answer generation by the LLM
        # Removed 'retrieved_metadata' from the prompt
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for question-answering tasks. ONLY use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and factual.\n\nContext: {context}"),
            ("user", "{query}")
        ])

        # Helper function to format the retrieved documents for the RAG prompt
        def format_docs(docs: List[Document]) -> str:
            """Formats document content into a string for the RAG prompt."""
            print("\n--- Retrieved Document Chunks (for RAG Context) ---")
            if not docs:
                print("No documents were retrieved for the given query.")
            for i, doc in enumerate(docs):
                print(f"--- Chunk {i+1} from Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')} ---")
                print(doc.page_content[:500] + "...") # Print first 500 chars for brevity
                print("-" * 30)
            return "\n\n".join([doc.page_content for doc in docs])

        # Get the retriever instance directly
        # --- IMPORTANT CHANGE: Increased k to retrieve more chunks ---
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 8}) # Retrieve 8 relevant chunks (was 4)

        # Construct the RAG chain using LangChain's RunnablePassthrough for simple data flow
        chain = (
                {
                    # 'context' key will be filled by the retriever's output (list of Documents)
                    "context": retriever | format_docs, # Pass retriever output to format_docs
                    # 'query' key will be the original input string, passed through
                    "query": RunnablePassthrough()
                }
                | rag_prompt # Pass the structured input to the prompt template
                | llm_model # Pass the templated prompt to the LLM for generation
                | StrOutputParser() # Parse the LLM's output to a string
        )

        # --- Example Question for Invocation ---
        questions = '''what were teslas automative sales in millions for 2024?'''
        result = chain.invoke(questions)
        print("\n--- Raw Chain Output ---")
        print(result)

        print(f"\n--- Invoking chain with question: {questions} ---")
        try:
            result = chain.invoke(questions)
            print("\n--- Raw Chain Output ---")
            print(result)

            # Display the result as Markdown if in an environment supporting IPython.display
            if 'IPython.display' in globals() and result:
                print("\n--- Formatted Markdown Output ---")
                display(Markdown(result))
            elif not result:
                print("\nChain returned an empty string.")

        except Exception as e:
            print(f"\n--- An error occurred during chain invocation: {e} ---")
            print("Please ensure Ollama server is running and the 'llama3:latest' model is pulled, "
                  "and 'nomic-embed-text' is also pulled and running.")


if __name__ == "__main__":
    l = llm()
    db_persist_path = 'data/vector_db/vector_db'
    tesla_doc_path = 'data/raw/tsla-20241231-gen.pdf' # Make sure this PDF path is correct

    # --- FOR DEBUGGING & REBUILDING: Uncomment to force recreation of DB with new chunking ---
    # It is crucial to uncomment this and run at least once to ensure the DB
    # is rebuilt with the new chunk_size (1000) for effective RAG.
    if os.path.exists(db_persist_path):
        print(f"Removing existing DB at {db_persist_path} to force recreation with new chunking strategy.")
        shutil.rmtree(db_persist_path)
    # --- END DEBUGGING / REBUILDING ---

    # Check if the vector database already exists and load it
    if l.load_vector_db(db_path=db_persist_path):
        print("Existing vector DB loaded. Skipping document loading and embedding.")
    else:
        # If DB does not exist, create it
        print("Creating new vector DB...")
        # Ensure the 'data/raw' directory exists and contains the PDF
        os.makedirs(os.path.dirname(tesla_doc_path), exist_ok=True)

        all_documents_for_db = []
        tesla_chunks = l.load_and_split_document(tesla_doc_path, "TESLA Financial Report")
        all_documents_for_db.extend(tesla_chunks)

        if all_documents_for_db: # Only store if there are documents to store
            l.store_in_vector_db(all_documents_for_db, db_path=db_persist_path)
            print("Vector DB created and loaded successfully.")
        else:
            print("No documents were loaded or split. Vector DB not created.")
            # If no documents, there's no DB to use for retrieval, so exit or handle
            exit("Exiting: No documents to process for vector DB.")

    # Proceed with the retriever chain if vector_db is available
    l.multiquery_retriever()
