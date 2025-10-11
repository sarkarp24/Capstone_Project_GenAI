from fastapi import FastAPI, HTTPException
from typing import Dict
import uvicorn
from capstone_project import rag_simple,rag_retriever, llm

# FastAPI Endpoint
app = FastAPI()
@app.post("/medical_assistance/")
async def analyze_gait(question: Dict):
    try:
        #print("Received question:", question['question'])
        response = await rag_simple(question['question'],rag_retriever,llm)
        #response = "Hello, this is a placeholder response."
        #print("Response from RAG model:", response)
        payload =  {
                "question": question,
                "answer": response
            }
        
        return payload
    except Exception as e:
        #print("Error occurred while processing the request:")
        #print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
uvicorn.run(app,host="localhost",port=8005)