📚 RAG Chatbot – FastAPI + LangChain + Groq + FAISS + HuggingFace

A production-ready Retrieval-Augmented Generation (RAG) chatbot that supports PDF/text uploads, document embedding, vector storage, and conversational Q&A using Groq LLMs, HuggingFace embeddings, and FAISS vector store, deployed with Docker + GitHub Actions CI/CD + AWS EC2/ECR.

🚀 Features

Upload PDF/Text files

Extract text & split into chunks

Generate embeddings using Sentence Transformers

Store & retrieve vectors using FAISS

Natural-language querying using Groq LLM (LLaMA / Mixtral)

FastAPI backend with HTML UI (Jinja templates)

Fully containerized with Docker

Automated CI/CD via GitHub Actions → ECR → EC2 deployment

Persistent uploads directory

🗂 Project Structure
app/
 ├── main.py                   # FastAPI entry point
 ├── config.py                 # App configuration / env vars
 ├── logger.py                 # Custom logger
 ├── templates/                # Jinja2 HTML templates
 ├── static/                   # CSS/JS assets
 ├── data_processing/
 │     └── data_processing.py  # File parsing + chunking
 ├── vector_store/
 │     └── vector_store.py     # FAISS + embeddings logic
uploads/                       # Uploaded PDF/text files
Dockerfile                     # Production container build
requirements.txt               # Python dependencies
.github/workflows/deploy.yml  # CI/CD workflow

⚙️ Requirements

Python 3.11

Docker

AWS account (ECR + EC2)

GitHub Actions runner (self-hosted on EC2 if using CD)

🔧 Installation (Local)
1️⃣ Clone the project
git clone https://github.com/<your-username>/<repo>.git
cd <repo>

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Run FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000


Visit:

http://localhost:8000

🐳 Running With Docker
Build image
docker build -t rag_bot:latest .

Run container
docker run --rm -it -p 8000:8000 rag_bot:latest


🌐 API Endpoints
GET /

Renders the home page UI (HTML).

POST /query

Send a user question:

{
  "query": "What is attention in transformers?"
}

POST /uploadfile/

Upload PDF / text files to index.

🔒 Environment Variables

Place in .env or pass via Docker/EC2 environment:

Variable	Description
OPENAI_API_KEY	Groq-compatible OpenAI key (Groq API)
GROQ_API_KEY	Direct Groq API key if using langchain_groq
AWS_ACCESS_KEY_ID	For ECR/EC2 deployment
AWS_SECRET_ACCESS_KEY	""
AWS_DEFAULT_REGION	AWS region
🚀 CI/CD Pipeline (GitHub Actions → AWS ECR → EC2)

The project includes a fully automated deployment workflow:

1. On push to main:

GitHub Actions builds Docker image

Pushes to AWS ECR

2. EC2 (self-hosted runner)

Pulls latest image

Stops previous container

Restarts new version automatically

Workflow file:

.github/workflows/deploy.yml

🛠 Troubleshooting
PDF upload error: pypdf missing

Fixed by adding to requirements:

pypdf

NumPy ABI mismatch (NumPy 2.x error)

Solution:

numpy<2

FAISS installation issues

Use CPU version:

faiss-cpu

Port not accessible externally

Check EC2 security group:

Inbound → allow TCP 8000

📌 Future Improvements

Add vector store persistence across container rebuilds

Add authentication

Frontend UI improvements

❤️ Credits

Built using:

FastAPI

LangChain

Groq LLM

Sentence Transformers

FAISS

Docker

AWS ECR/EC2
