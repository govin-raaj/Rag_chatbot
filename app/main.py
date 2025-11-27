from langchain_core.messages import BaseMessage, SystemMessage,HumanMessage
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from app.data_processing.data_processing import ProcessData
from app.vector_store.vector_store import VectorStore
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from fastapi.templating import Jinja2Templates
from langchain_core.tools import tool
from typing import TypedDict, Annotated,List 
from langgraph.graph import StateGraph, START
from fastapi.params import Form as form
from langchain_groq import ChatGroq
from app.config import Config
from app.logger import logging
import os
import uvicorn
import pathlib


app = FastAPI()

vector_store = VectorStore()
logging.info("Vector store initialized.")

templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "uploads"
logging.info(f"Upload directory set to: {UPLOAD_DIR}")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    logging.info(f"Sanitizing filename: {filename}")
    return pathlib.Path(filename).name

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    logging.info(f"Received file upload request: {file.filename}")
    if not file or not file.filename:
        logging.error("No file selected for upload.")
        return JSONResponse({"error": "No file selected"}, status_code=400)

    filename = sanitize_filename(file.filename)
    logging.info(f"Sanitized filename: {filename}")

    if not filename.lower().endswith((".txt", ".pdf")):
        logging.error("Unsupported file type uploaded.")
        return JSONResponse(
            {"error": "Only .txt and .pdf files are supported"},
            status_code=400
        )

    logging.info(f"Saving uploaded file to {UPLOAD_DIR}/{filename}")
    upload_path = os.path.join(UPLOAD_DIR, filename)

 
    try:
        contents = await file.read()  
        with open(upload_path, "wb") as f:
            f.write(contents)
            logging.info(f"File {filename} saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        logging.info("Closing uploaded file.")
        await file.close()


    try:
        processor = ProcessData(upload_path)
        logging.info(f"Processing document: {upload_path}")
        chunks = processor.process_document()
        logging.info(f"Document processed into {len(chunks)} chunks.")
    except Exception as e:
        logging.error(f"Error processing document: {e}")

        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


    try:
        logging.info(f"Adding {len(chunks)} chunks to vector store.")
        vector_store.add_documents(chunks)
    except Exception as e:
        logging.error(f"Error adding documents to vector store: {e}")
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error adding documents: {e}")


    try:
        logging.info(f"Removing uploaded file: {upload_path}")
        if os.path.exists(upload_path):
            os.remove(upload_path)
    except Exception:
       
        pass

    return JSONResponse(
        {"message": "File processed successfully", "chunks": len(chunks)},
        status_code=200
    )

@app.get("/", response_class=HTMLResponse) 
async def home(request: Request): 
    logging.info("Rendering home page.")
    return templates.TemplateResponse("index.html", {"request": request})  

@app.post("/query")
async def query_rag(query:str= form(...)):
    try:
        logging.info(f"Received query: {query}")
        # llm_service=LLmService(vector_store,query)
        logging.info("Generating response from LLM service.")

        initial_messages = [HumanMessage(content=query)]
        initial_state = {"messages": initial_messages}
        CONFIG = {"configurable": {"thread_id": "thread_1"}}
        response= chatbot.invoke(initial_state,config=CONFIG)
        ai_message = response['messages'][-1].content
        return JSONResponse({"response": ai_message}, status_code=200)
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
    



llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=Config.qroq_api_key,
        )

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



@tool
def retrieval(query: str) -> List[dict]:
    """
    Retrieve relevant information from the uploaded PDF(s).
    """
    if vector_store.vector_store is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    try:
        retr = vector_store.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retr.invoke(query)  
    except Exception:

        docs = vector_store.similarity_search(query, k=3)

    context = [getattr(d, "page_content", str(d)) for d in docs]
    metadata = [getattr(d, "metadata", {}) for d in docs]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
    }


tools=[retrieval]

llm_with_tool=llm.bind_tools(tools)

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    system_message = SystemMessage(
        content=(
        "You are a helpful assistant. When the user asks about the uploaded document, "
        "use the retriever tool to fetch relevant document passages. "
        "The retrieval tool returns a list of objects with 'page_content' and 'metadata'. "
        "If there are no documents available, ask the user to upload the PDF. "
        "If you genuinely do not know the answer, say you don't know."
        )
    )

    messages = [system_message,*state['messages']]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)