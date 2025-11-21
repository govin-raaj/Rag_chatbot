from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from src.data_processing.data_processing import ProcessData
from src.vector_store.vector_store import VectorStore
from src.services.llm_service import LLmService
import os
import pathlib

app = FastAPI()

vector_store = VectorStore()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    # Simple sanitize: keep only the basename (remove any path components)
    return pathlib.Path(filename).name

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Basic validation
    if not file or not file.filename:
        return JSONResponse({"error": "No file selected"}, status_code=400)

    filename = sanitize_filename(file.filename)
    if not filename.lower().endswith((".txt", ".pdf")):
        return JSONResponse(
            {"error": "Only .txt and .pdf files are supported"},
            status_code=400
        )

    upload_path = os.path.join(UPLOAD_DIR, filename)

    # Save upload to disk (async-safe)
    try:
        contents = await file.read()  # read the whole uploaded file
        with open(upload_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await file.close()

    # Process file into chunks
    try:
        processor = ProcessData(upload_path)
        chunks = processor.process_document()
    except Exception as e:
        # ensure we remove the saved file on processing failure
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    # Add to vector store (you must have vector_store available in scope)
    try:
        # vector_store.add_documents expects a list of Documents (langchain Documents)
        vector_store.add_documents(chunks)
    except Exception as e:
        # cleanup saved file if desired
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error adding documents: {e}")

    # optionally remove the saved file after processing & storing
    try:
        if os.path.exists(upload_path):
            os.remove(upload_path)
    except Exception:
        # ignore cleanup errors
        pass

    return JSONResponse(
        {"message": "File processed successfully", "chunks": len(chunks)},
        status_code=200
    )

@app.get("/")
def home():
    return JSONResponse({'message': 'Welcome to the RAG Chatbot API'}), 200
    

@app.post("/query")
async def query_rag(query:str):
    try:
        llm_service=LLmService(vector_store,query)
        response=llm_service.generate_response()
        return JSONResponse({"response": response}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
    

@app.get("/vectorstore_result")
def vectorstore_result(query: str):
    try:
        results = vector_store.similarity_search(query, k=4)
        formatted_results = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]
        return JSONResponse({"results": formatted_results}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during similarity search: {e}")
    