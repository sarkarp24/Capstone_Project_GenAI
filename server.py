from fastapi import FastAPI, HTTPException
from typing import Dict
import uvicorn
from capstone_project import AdvancedRAGPipeline,rag_retriever, llm

# FastAPI Endpoint
app = FastAPI()
@app.post("/medical_assistance/")
async def analyze_gait(question: Dict):
    try:
        print("Received question:", question['question'])
        adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
        response = await adv_rag.query(question['question'],top_k=5, min_score=0.0, stream=True, summarize=True)
        #response = "Hello, this is a placeholder response."
        #print("Response from RAG model:", response)
        payload =  {
                "question": question,
                "answer": response['answer']
            }
        
        return payload
    except Exception as e:
        #print("Error occurred while processing the request:")
        #print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
uvicorn.run(app,host="localhost",port=8005)