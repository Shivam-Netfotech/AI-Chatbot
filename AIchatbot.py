from fastapi import FastAPI, HTTPException
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import os

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "OPENAI KEY" 
embeddings = OpenAIEmbeddings()

pdfreader = PdfReader("PDF-location")

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

chat_history = []

@app.post("/chatbot")
async def chatbot_endpoint(user_input: str):
    if user_input.lower() == 'exit':
        raise HTTPException(status_code=200, detail="Exiting chatbot.")

    docs = document_search.similarity_search(user_input)
    result = chain({"question": user_input, "chat_history": chat_history, "input_documents": docs, "temperature": 0.5})
    response = result.get('output_text', 'No answer found')
    chat_history.append((user_input, response))

    return {"response": response}
