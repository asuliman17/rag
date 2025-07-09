A python script where I a rag model for Q&A against Tesla's 2024 10k. 

Steps taken: 
- split the data into chunks
  - utilzied overlapping
  - added metadata to those chunks
- stored chunks into vector_db using chroma db
- Converted chunks into embeddings using nomic-embed-text model.
- Ran llama3 locally to use embeddings to answer questions from the user.  
