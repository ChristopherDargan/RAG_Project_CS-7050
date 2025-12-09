# RAG_Project_CS-7050
RAG-Based Document Retrieval System for Technical Documentation 
# Project Summary:
The idea of the project is to implement a Retrieval-Augmented Generation
(RAG) system that uses semantic search to retrieve relevant documentation chunks from
technical documents. The project will focus on the “Retrieval” aspect of RAG as it pertains the
most to data mining / document retrieval. The system will use efficient nearest-neighbor searches
on document embeddings allowing a user to retrieval relevant context based on the initial query.
# Dataset Description:
For the dataset I will use the scikit-learn documentation which is
approximately 93 MB of technical documentation including API references, user guides,
tutorials, examples, etc. The documents will be chunked into meaningful segments. Each chunk
is then converted into a higher dimensional embedding vector using a sentence transformer
model / generic embedding model.
# Data Mining Algorithm:
The proposed fundamental data mining algorithm being used is a knearest neighbor (kNN) search. Each segmented chunk of text will be transformed into a high
dimensional vector representation of its original text. Given a user’s query we can computing the
Euclidean distance / cosine similarity against all of our embedded document embedding,
returning the top k similar chunks (returns relevant data to the question). In turn this will allow
the user to efficiently search documents across an entire document space. 
