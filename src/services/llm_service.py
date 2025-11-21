from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.config import Config
from langchain.schema import Document
from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from src.vector_store.vector_store import VectorStore
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

vector_store = VectorStore(path="faiss_store")
retriever = vector_store.vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store.vector_store else None


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_KEY,
    max_new_tokens=200,
)

model = ChatHuggingFace(llm=llm)



class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool
    messages: Annotated[List[dict], add_messages]


def decide_retrieval(state: AgentState) -> AgentState:
    msgs = state.get("messages", []) or []
    q = state["question"].lower()
    keywords = ["what", "how", "explain", "describe", "tell me"]
    needs = any(k in q for k in keywords)
    return {**state, "needs_retrieval": needs, "messages": [{"type":"system","content":f"decide_retrieval -> needs_retrieval={needs}"}]}


def retrieve_documents(state: AgentState) -> AgentState:
    _ = state.get("messages", []) or []

    if retriever is None:
        return {**state, "documents": [], "messages": [{"type":"system","content":"No retriever configured"}]}

    query = state.get("question","") or ""
    if not query.strip():
        return {**state, "documents": [], "messages": [{"type":"system","content":"Empty query; skipping retrieval"}]}

    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
        else:
            docs = retriever(query)
    except Exception as e:
        return {**state, "documents": [], "messages": [{"type":"system","content":f"Retrieval error: {e}"}]}

    if not docs:
        return {**state, "documents": [], "messages": [{"type":"system","content":"No documents retrieved"}]}

    return {**state, "documents": docs, "messages": [{"type":"system","content":f"Retrieved {len(docs)} documents"}]}


def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("documents", []) or []

    if docs:
        context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
        final_prompt = f"""
        Based on the following context, answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    else:
        final_prompt = f"Answer the following question:\n{question}"

    # Call model defensively (try common methods)
    try:
        resp = model.invoke(final_prompt)
    except Exception:
        try:
            resp = model.generate(final_prompt)
        except Exception:
            try:
                resp = model(final_prompt)
            except Exception as e:
                # If all attempts fail, store the error as the answer
                return {**state, "answer": f"Model error: {e}", "messages": [{"type": "system", "content": f"Model invocation failed: {e}"}]}

    answer_text = _extract_model_text(resp)

    # Append the human question and assistant answer so InMemorySaver + add_messages maintain history
    return {**state, "answer": answer_text, "messages": [
        {"type": "human", "content": question},
        {"type": "assistant", "content": answer_text},
    ]}

def _extract_model_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return getattr(resp, "content")
    if hasattr(resp, "text"):
        return getattr(resp, "text")
    if hasattr(resp, "generated_text"):
        return getattr(resp, "generated_text")
    if isinstance(resp, dict):
        for k in ("content", "text", "generated_text", "output", "result"):
            if k in resp:
                return resp[k]
        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            c0 = resp["choices"][0]
            if isinstance(c0, dict) and "text" in c0:
                return c0["text"]
    return str(resp)


def should_retrieve(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


checkpointer = InMemorySaver()

workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("decide")

workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {"retrieve": "retrieve", "generate": "generate"}
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile(checkpointer=checkpointer)

