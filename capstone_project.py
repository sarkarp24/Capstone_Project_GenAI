###Document Structure

from langchain_core.documents import Document
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
import jq
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.llms import ollama
import os
from dotenv import load_dotenv
import json
load_dotenv()

loader_kwargs = {
    # .[] is used to iterate over every item in the root array
    "jq_schema": ".[]",
    
    # Specifies the key containing the main text content
    "content_keys_path": "content", 
    
    # List of keys to pull as metadata for the resulting Document object
    "metadata_keys_path": [
        "id",
        "title",
        "source",
        "date_generated"
    ],
}
#print(os.getcwd()+"/data/json/synthetic_medical_kb.json")
loader = JSONLoader(
    file_path='data/json/synthetic_medical_kb.json',
    jq_schema=loader_kwargs["jq_schema"],
    text_content=False)

print(loader)

json_documents=loader.load()
#print(f"Loaded {len(json_documents)} documents.")
#json_documents


for doc in json_documents:
    dict_doc = json.loads(doc.page_content)
    joint_content = '\n'.join(f"{d}:{dict_doc[d]}" for d in dict_doc)
    #print(joint_content)
    doc.page_content = joint_content
    #print("-----")
#print(loader)
'''
loader = JSONLoader(
    file_path='../data/json/synthetic_medical_kb.json',
    jq_schema=loader_kwargs["jq_schema"],
    text_content=False)

'''
json_documents=loader.load()
#print(f"Loaded {len(json_documents)} documents.")
'''
for doc in json_documents:
    print(doc)
    print(doc.page_content)
    print("-----")
'''
#json_documents

def split_documents(documents,chunk_size=1600,chunk_overlap=300):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    

    return split_docs

chunks=split_documents(json_documents)

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): 
    #def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully. Embedding dimension:", self.model.get_sentence_embedding_dimension())
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        if not self.model:
            raise ValueError("Model is not loaded.")
        
        print(f"Generating Embedding for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Embeddings generated with shape:", embeddings.shape)
        return embeddings
    

embedding_manager=EmbeddingManager()
print(embedding_manager)

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "json_documents", persist_directory: str = "data/capstone_project/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # print(metadata)
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore=VectorStore()
print(vectorstore)

'''
texts = [doc.page_content for doc in chunks]
print(len(texts))
for i in range(0, len(texts), 5000):
    batch_texts = texts[i:i+5000]
    embeddings = embedding_manager.generate_embeddings(batch_texts)
    vectorstore.add_documents(chunks[i:i+5000], embeddings)
#embeddings = embedding_manager.generate_embeddings(texts)
#vectorstore.add_documents(chunks, embeddings)
'''

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)
#print(rag_retriever)



### Initialize the ollama LLM
llm=ollama.Ollama(model="llama3.2",temperature=0.1)

## 2. Simple RAG function: retrieve context + generate response
class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    async def query(self, question: str, top_k: int = 5, min_score: float = 0.0, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        emergency = ['emergency', 'urgent', 'immediate help', 'as soon as possible', 'right away', 'critical', 'danger', 'threat','bleeding','accident','injury',
                     'chest pain','chest discomfort','shortness of breath','severe pain','unconscious','unresponsive','dizziness','confusion','seizure','pregnant','suicide','suicidal']
        
        #print(question)
        if any(word in question.lower() for word in emergency):
            warning_message = "WARNING!!! If this is a medical emergency, please call 911 or go to the nearest emergency care immediately."
            print(warning_message)
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        #print(f"Retrieved {(results)} documents for the query.")
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            
            print(context)
            for s in sources:
                print(s['score'])
            
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            '''
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            '''
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response
            #print("LLM Response:", answer)

        if answer == 'No relevant context found.':
            answer = None

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        #print("Citations:", citations)
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        #print('answer of the query:',answer)
        if summarize and answer:
            summary_prompt = f"Write the warning message based on the answer:\n{answer}"
            #print('summary_prompt:',summary_prompt)
            summary_resp = self.llm.invoke([summary_prompt])
            #print("Summary Response:", summary_resp)
            summary = summary_resp

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'warning': summary,
            'history': self.history
        }
