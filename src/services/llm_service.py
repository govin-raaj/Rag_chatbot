from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

class LLmService:
    def __init__(self,vector_store,query: str):
        self.llm= HuggingFaceEndpoint(
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


    def query_refinement(self) -> str:
        query_prompt=PromptTemplate(
            template="""Improve the clarity, grammar, and structure of the following user query while  \
            keeping the original intent unchanged. Remove ambiguity and make it easy for an LLM to understand:" \
            User Query: {query}
            """,
            input_variables=["query"]
            )   
        parser=StrOutputParser()

        chain= query_prompt | self.model | parser 

        refined_query=chain.invoke({"query":self.query})

        return refined_query
    

    def get_documents(self) -> List[Document]:
        query=self.query_refinement()

        if self.retriever is None:
            return []   # No documents available → RAG fallback

        try:
            results = self.retriever.get_relevant_documents(query)
        except Exception:
            results = []

        return results
    

    
    def generate_response(self) -> str:
        documents = self.get_documents()

        if not documents:
            context = "No external documents provided. Answer only from general knowledge."
        else:
            context = "\n\n".join([doc.page_content for doc in documents])
        
        query=self.query_refinement()

        prompt = PromptTemplate(
            template="""You are an AI assistant that provides helpful and accurate information based on the provided context. \
            Use the context to answer the question as accurately as possible. If the context does not contain relevant information, \
            respond with 'I don't know'.

            Context:
            {context}

            Question:
            {query}

            Answer:""",
            input_variables=["context", "query"]
        )
        
        parser = StrOutputParser()

        chain = prompt | self.model | parser

        response = chain.invoke({"context": context, "query": query})
    
        return response



    


