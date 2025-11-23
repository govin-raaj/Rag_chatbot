from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class LLmService:
    def __init__(self, vector_store, query: str):
        self.llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="text-generation"
        )

        self.model = ChatHuggingFace(llm=self.llm)
        self.query = query
        self.vector_store = vector_store

        self.retriever = None
        try:
            if vector_store and hasattr(vector_store, "vector_store"):
                self.retriever = vector_store.vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
        except Exception:
            self.retriever = None

        # only ConversationBufferMemory as requested
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    def query_refinement(self) -> str:
        query_prompt = PromptTemplate(
            template="""Improve the clarity, grammar, and structure of the following user query while
keeping the original intent unchanged. Remove ambiguity and make it easy for an LLM to understand:

User Query: {query}
""",
            input_variables=["query"]
        )
        parser = StrOutputParser()
        chain = query_prompt | self.model | parser
        refined_query = chain.invoke({"query": self.query})
        return refined_query

    def get_documents(self) -> List[Document]:
        query = self.query_refinement()
        if self.retriever is None:
            return []   
        try:
            results = self.retriever.invoke(query)
        except Exception:
            results = []
        return results

    def _build_memory_text(self) -> str:
        """Create a simple text snapshot of the conversation buffer for the prompt."""
        memory_text = ""
        try:
            chat_mem = getattr(self.memory, "chat_memory", None)
            if chat_mem and hasattr(chat_mem, "messages"):
                for msg in chat_mem.messages:
                  
                    typ = getattr(msg, "type", None)
                    content = getattr(msg, "content", str(msg))
                    if typ == "human" or typ == "user":
                        memory_text += f"User: {content}\n"
                    else:
                        memory_text += f"AI: {content}\n"
        except Exception:
            
            memory_text = ""
        return memory_text

    def generate_response(self) -> str:
  
        documents = self.get_documents()
        if not documents:
            context = "No external documents provided. Answer only from general knowledge."
        else:
            context = "\n\n".join([doc.page_content for doc in documents])

        query = self.query_refinement()


        if self.retriever is not None:
            conv_chain = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                retriever=self.retriever,
                memory=self.memory,
                output_key="answer"
            )
    
            result = conv_chain.invoke({"question": query})
            answer = result.get("answer") or result.get("output_text") or str(result)


            try:
                self.memory.save_context({"question": query}, {"answer": answer})
            except Exception:
               
                pass

            return answer

    
        memory_text = self._build_memory_text()

        prompt_template = PromptTemplate(
            template="""You are an AI assistant that provides helpful and accurate information based on the provided context and known user memory.
                Use the context and any stored memory facts to answer the question as accurately as possible. If the context does not contain relevant information, respond with "I don't know".

                    Context:
                    {context}

                    Memory:
                    {memory}

                    Question:
                    {query}

                    Answer:""",
            input_variables=["context", "memory", "query"]
        )

        parser = StrOutputParser()
        chain = prompt_template | self.model | parser

        response = chain.invoke({
            "context": context,
            "memory": memory_text,
            "query": query
        })

  
        try:
            self.memory.save_context({"question": query}, {"answer": response})
        except Exception:
            pass

        return response
