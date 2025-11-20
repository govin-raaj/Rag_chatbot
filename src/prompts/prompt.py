from langchain_core.prompts import ChatPromptTemplate

class prompt:
    
    template = """
        You are an helpful assistant for question-answering tasks.
        Use the following pieces of retrieved context and chat history to answer the question.

        Chat history:
        {chat_history}

        Question: {question}
        Context: {context}
        Answer:
        """
    qa_prompt =ChatPromptTemplate.from_template(template)