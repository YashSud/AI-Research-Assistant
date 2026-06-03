import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Enterprise AI Research Assistant API",
    description="FastAPI Backend for RAG Document Processing & Citations reasoning",
    version="1.0.0"
)

# Enable CORS for frontend dashboard queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local Storage Directory Paths
UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize RAG Ingestion Pipeline
pipeline = RAGPipeline(persist_directory="./vector_db")

class QueryRequest(BaseModel):
    query: str = Field(..., example="What was the EBITDA margin in Q3?", min_length=3, max_length=1000)
    collection: str = Field("default", example="finance-2026")

class QueryResponse(BaseModel):
    answer: str
    citations: list
    scores: dict

@app.post("/api/v1/upload", tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    # Receives PDF bytes, stores locally, and triggers semantic vector indexing
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in the ingestion pipeline.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger Ingestion
        num_chunks = pipeline.ingest_document(file_path)
        return {
            "status": "success",
            "file_name": file.filename,
            "chunks_created": num_chunks,
            "message": f"Document successfully ingested into ChromaDB. Split into {num_chunks} semantic chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF ingestion: {str(e)}")

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Inference"])
async def query_pipeline(request: QueryRequest):
    # Executes semantic retrieval and returns citation-grounded output
    try:
        response = pipeline.execute_query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@app.get("/api/v1/status", tags=["Observability"])
async def system_status():
    # Returns vector database collections sizes and directory stats
    db_size = 0
    if os.path.exists("./vector_db"):
        for root, dirs, files in os.walk("./vector_db"):
            db_size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
            
    num_files = 0
    if os.path.exists(UPLOAD_DIR):
        num_files = len([f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))])
        
    return {
        "status": "online",
        "vector_store": "ChromaDB Persistent Index",
        "files_ingested": num_files,
        "index_disk_bytes": db_size,
        "active_model": "gpt-4o-2024-05-13"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
