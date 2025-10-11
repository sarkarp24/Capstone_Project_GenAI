# Ignore all the _bkp files
# Pre-requisite
    1.  Install Ollama model first and then start the service
        Command to start the service: ollama run llama3.2

# Download the project from the GitHub and configure locally on your machine 

# Instruction to run .ipynyb
    1.  Run all the cells sequentially from capstone_project.ipynyb

# Instruction to run as API
    1.  Open server.py, open a terminal and change the directory where server.py is located
    2.  Run the command 'python server.py' or 'python3 server.py' based on the version of the python installed on the machine. It should return following message on the terminal.
    
            prodyotsarkar@Prodyots-MacBook-Air Capstone_Project_GenAI % python3 server.py        
            Split 10500 documents into 10500 chunks
            Loading model: all-MiniLM-L6-v2
            Model loaded successfully. Embedding dimension: 384
            <capstone_project.EmbeddingManager object at 0x110b8c980>
            Vector store initialized. Collection: json_documents
            Existing documents in collection: 73500
            <capstone_project.VectorStore object at 0x151f8dbe0>
            <capstone_project.RAGRetriever object at 0x151f8fcb0>
            /Users/prodyotsarkar/Documents/GenAI/Capstone_Project_GenAI/capstone_project.py:313: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
            llm=ollama.Ollama(model="llama3.2",temperature=0.1)
            INFO:     Started server process [18552]
            INFO:     Waiting for application startup.
            INFO:     Application startup complete.
            INFO:     Uvicorn running on http://localhost:8005 (Press CTRL+C to quit)
    3.  Open client.py, open another terminal and change the directory where client.py is located
    4.  Run the command 'python client.py' or 'python3 client.py' based on the version of the python installed on the machine.
    5.  The API should answer based on the question asked in client.py.

        Following is the response from API of the question "What is Rheumatoid Arthritis and what are the treatments available?".
        prodyotsarkar@Prodyots-MacBook-Air Capstone_Project_GenAI % python3 client.py               
        Rheumatoid Arthritis (RA) is a chronic autoimmune disease characterized by joint inflammation, pain, and stiffness. The primary symptoms include polydipsia (increased thirst), chest pain, and recurrent cough.

        The standard treatment for RA includes:

        1. Physical therapy, often guided by a Cardiology specialist.
        2. Regular follow-up appointments (quarterly or semi-annually) to monitor disease progression and adjust pharmacological therapy.
        3. Patient education is crucial for successful long-term management.

        Note: The question does not mention specific medications as treatments, but rather emphasizes the importance of physical therapy, regular check-ups, and patient education in managing RA.




