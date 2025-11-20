from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from src.data_processing.data_processing import process_data
from src.vector_store.vector_store import VectorStore
from src.services.llm_service import rag_app

app = FastAPI()

vector_store = VectorStore()


@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    try:
        
        if 'file' not in Request.files:
            return JSONResponse({'error': 'No file provided'}), 400
        
        file = Request.files['file']
        if file.filename == '':
           
            return JSONResponse({'error': 'No file selected'}), 400

        # Check file extension
        if not file.filename.endswith(('.txt', '.pdf')):
            return JSONResponse({'error': 'Only .txt and .pdf files are supported'}), 400
        

        try:
            file_processor = process_data(file)
            chunks = file_processor.process_document()

        except Exception as e:
            raise HTTPException(500, "Error processing file") from e
        
        try:
            vector_store.add_documents(chunks)
            return JSONResponse({'message': 'File processed and data added to vector store successfully'}), 200
        
        except Exception as e:
            raise HTTPException(500, "Error adding documents to vector store") from e
        

        return JSONResponse({'message': 'File uploaded successfully', 'no of chunks':len(chunks)}), 200

        
    except Exception as e:  
        raise HTTPException(500, "Internal server error") from e
    


@app.get("/")
def home():
    return JSONResponse({'message': 'Welcome to the RAG Chatbot API'}), 200
    

@app.post("/query")
async def query_rag(payload: dict):
    try:
        question = payload.get("question", "")

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        # Run LangGraph RAG agent
        result = rag_app.invoke({"question": question})

        return {"answer": result["answer"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    