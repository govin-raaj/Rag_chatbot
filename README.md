# 🤖 RAG Chatbot – FastAPI | LangChain | Groq | HuggingFace | FAISS

An end-to-end RAG (Retrieval-Augmented Generation) system demonstrating modern AI engineering practices — from document ingestion, text chunking, embedding generation, vector storage, semantic search, conversational response generation, Dockerization, CI/CD, to AWS EC2 deployment.

This project is designed not only as a functional RAG system, but as a portfolio showcase demonstrating:

✅ **AI Engineering** (LLM orchestration, embedding models, vector search)  
✅ **Backend Development** (FastAPI, routing, file handling, HTML templating)  
✅ **MLOps / LLMOps** (CI/CD, ECR push, EC2 deployment, Docker)  
✅ **Cloud & DevOps** (AWS EC2, ECR, IAM, GitHub Actions runners)  
✅ **Software Engineering** (logging, modular structure, environment config)

---

## 📂 Project Structure
```
.
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── config.py                   # Environment & API config
│   ├── logger.py                   # Application-wide logging
│   ├── templates/                  # Jinja2 HTML templates
│   ├── static/                     # CSS/JS frontend assets
│   ├── data_processing/
│   │     └── data_processing.py    # PDF/Text loading + text splitting
│   ├── vector_store/
│   │     └── vector_store.py       # FAISS and embeddings management
│
├── uploads/                        # Uploaded user files
├── Dockerfile                      # Docker build configuration
├── requirements.txt                # Python dependencies
├── .github/workflows/deploy.yml    # CI/CD Pipeline
└── README.md
```


## 📄 Document Ingestion Pipeline

 **This project supports PDF and text file ingestion:**
 **✔ Extraction & Processing Steps**
* Upload PDF/TXT files via UI
* Extract raw text
* Apply RecursiveCharacterTextSplitter
* Generate embeddings using:
* sentence-transformers/all-MiniLM-L6-v2
* Store embeddings in FAISS vector store
* Automatically retrieve relevant chunks during user queries



## 🔍 Conversational Retrieval Chain

**The chatbot uses:**
* Groq LLM (LLaMA 3, Mixtral) for response generation
* LangChain for routing messages, tool composition, and state graph
* FAISS for relevant passage retrieval



## 📝 Logging & Error Handling

**Integrated logging supports:**
* File upload logging
* PDF parsing failures
* Embedding pipeline tracking
* Query tracing and LLM response logging




## ☁️ AWS Deployment (CI/CD)

**Your project includes a complete CI/CD pipeline using GitHub Actions.**

**Workflow tasks:**
* Build Docker image
* Push to AWS ECR
* Trigger deployment job
* EC2 (self-hosted runner) pulls latest image
* Restarts container with new version


## 🔑 Required GitHub Secrets

* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_DEFAULT_REGION
* ECR_REPO
* GROQ_API_KEY




## 📊 Features Summary
* ✅ RAG-based Question Answering
* ✅ PDF/Text file uploads
* ✅ FAISS-based semantic search
* ✅ Groq-powered LLM responses
* ✅ Modular architecture
* ✅ Dockerized backend
* ✅ GitHub Actions CI/CD
* ✅ AWS EC2 deploymen* 







