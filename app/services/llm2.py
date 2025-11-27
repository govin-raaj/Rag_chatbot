from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,List 
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from app.config import Config


llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=Config.qroq_api_key,
        )

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



@tool
def retrieval(query: str) -> List[dict]:
    """
    Retrieve relevant docs from the vector store and return JSON-serializable results.
    Returns a list of dicts: [{"page_content": "...", "metadata": {...}}, ...]
    """

    retriever = _vs.as_retriever()

    try:
        docs: List[Document] = retriever.get_relevant_documents(query)
    except AttributeError:
 
        docs = retriever.invoke(query)

  
    results = []
    for d in docs:
    
        results.append({
            "page_content": getattr(d, "page_content", str(d)),
            "metadata": getattr(d, "metadata", {}) or {}
        })
    return results

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


class LLMSERVICE:
    def __init__(self,vectorstore,query):
        self.vector_store=vectorstore
        self.query=query
        