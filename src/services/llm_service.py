from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.config import Config
from langchain.schema import Document
from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from src.vector_store.vector_store import VectorStore


vector_store = VectorStore(path="faiss_store")
retriever = vector_store.vector_store.as_retriever() if vector_store.vector_store else None


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
    messages: Annotated[List, add_messages]


def decide_retrieval(state: AgentState) -> AgentState:
    q = state["question"].lower()
    keywords = ["what", "how", "explain", "describe", "tell me"]
    return {**state, "needs_retrieval": any(k in q for k in keywords)}


def retrieve_documents(state: AgentState) -> AgentState:
    if retriever is None:
        return {**state, "documents": []}

    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("documents", [])

    if docs:
        context = "\n\n".join(d.page_content for d in docs)
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

    resp = model.invoke(final_prompt)
    return {**state, "answer": resp.content}


def should_retrieve(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


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

rag_app = workflow.compile()
