from flask import Flask, request, jsonify, render_template
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.prompt import *
import os

app = Flask(__name__)

print("[INFO] Starting Flask app initialization...")
app = Flask(__name__)

print("[INFO] Loading environment variables from .env file...")
load_dotenv()


PINECONE_KEY = os.environ.get("PINECONE_KEY")
if not PINECONE_KEY:
    raise ValueError("PINECONE_KEY environment variable not set.")

OPENAI_KEY = os.environ.get("OPENAI_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_KEY environment variable not set.")

os.environ["PINECONE_API_KEY"] = PINECONE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

print("[INFO] Initializing HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()
print("[INFO] Embeddings initialized.")


index_name = "test"

# Embed each chunk and vector embeddings into the Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)