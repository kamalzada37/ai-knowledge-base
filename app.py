import streamlit as st  
import os  
import requests  
from bs4 import BeautifulSoup  
import chromadb  
from langchain_ollama import OllamaEmbeddings  
import ollama  
import re  
from duckduckgo_search import DDGS  
from wordcloud import WordCloud  
import matplotlib.pyplot as plt  
import fitz  # PyMuPDF for PDF processing  
from langchain.text_splitter import RecursiveCharacterTextSplitter  

LLM_MODEL = "llama3.2"  
CHROMA_DB_PATH = "chroma_db"  
COLLECTION_NAME = "knowledge_base"  

def setup_page():  
    st.set_page_config(page_title="AI Knowledge Base", layout="wide")  
    st.title("🤖 AI-Powered Knowledge Base")  

def initialize_chromadb():  
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)  

if "chat_history" not in st.session_state:  
    st.session_state.chat_history = []  
if "file_uploaded" not in st.session_state:  
    st.session_state.file_uploaded = False  
if "file_text" not in st.session_state:  
    st.session_state.file_text = ""  

class EmbeddingWrapper:  
    def __init__(self, embed_func):  
        self.embed_func = embed_func  
    def __call__(self, input):  
        return self.embed_func.embed_documents(input)  

embedding_function = OllamaEmbeddings(model=LLM_MODEL, base_url="http://localhost:11434")  
embed_wrapper = EmbeddingWrapper(embedding_function)  
collection = initialize_chromadb().get_or_create_collection(  
    name=COLLECTION_NAME, embedding_function=embed_wrapper  
)  

def sanitize_text(text):  
    swear_words = ["badword1", "badword2", "badword3"]  
    pattern = re.compile(r'\b(' + '|'.join(swear_words) + r')\b', re.IGNORECASE)  
    return pattern.sub("****", text)  

def search_web(query):  
    with DDGS() as ddgs:  
        return [result['href'] for result in ddgs.text(query, max_results=3)]  

def scrape_content(url):  
    headers = {"User-Agent": "Mozilla/5.0"}  
    response = requests.get(url, headers=headers)  
    if response.status_code != 200:  
        return ""  
    soup = BeautifulSoup(response.content, "html.parser")  
    return " ".join([p.get_text() for p in soup.find_all("p")])  

def process_and_store_web_documents(query):  
    urls = search_web(query)  
    documents = [scrape_content(url) for url in urls if url]  
    for i, doc in enumerate(documents):  
        if doc:  
            collection.add(ids=[f"doc_{i}"], documents=[doc], metadatas=[{"source": urls[i]}])  
    return documents, urls  

def query_ollama(query, context):  
    if not context.strip():  
        return "I couldn't find any relevant document content."  
    sanitized_query = sanitize_text(query)  
    prompt = [{  
        "role": "user",  
        "content": f"Use the following document content to answer:\n\n{context}\n\nQuestion: {sanitized_query}\nAnswer:"  
    }]  
    return ollama.chat(model=LLM_MODEL, messages=prompt)['message']['content']  

def process_uploaded_file(uploaded_file):  
    text = ""  
    if uploaded_file.name.endswith(".pdf"):  
        file_bytes = uploaded_file.read()  
        doc = fitz.open(stream=file_bytes, filetype="pdf")  
        for page in doc:  
            text += page.get_text("text")  
    else:  
        text = uploaded_file.getvalue().decode("utf-8")  
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  
    chunks = text_splitter.split_text(text)  
    
    if not chunks:  
        st.error("Failed to extract text from the document.")  
    
    for i, chunk in enumerate(chunks):  
        collection.add(ids=[f"file_chunk_{i}"], documents=[chunk], metadatas=[{"source": uploaded_file.name}])  
    
    return text  

def generate_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def main():  
    setup_page()  
    
    st.sidebar.header("Chat History")  
    if st.sidebar.button("Clear History"):  
        st.session_state.chat_history = []  
    for entry in st.session_state.chat_history:  
        st.sidebar.write(f"**Q:** {entry['query']}")  
        st.sidebar.write(f"**A:** {entry['response']}")  
    
    st.subheader("📂 Upload Documents")  
    uploaded_file = st.file_uploader("Upload PDFs or text files", type=["pdf", "txt"])  
    if uploaded_file:  
        if st.button("Process File"):  
            file_text = process_uploaded_file(uploaded_file)  
            st.session_state.file_uploaded = True  
            st.session_state.file_text = file_text   
            st.success("Document added to the knowledge base!")  
            with st.expander("Uploaded Document Preview"):  
                st.write(file_text[:1000] + "...")  
    
    st.subheader("🔍 Ask a Question")  
    query = st.text_input("Enter your query here:")  
    
    if st.button("Search & Analyze"):  
        with st.spinner("Fetching information..."):  
            context = ""  
            if st.session_state.get("file_uploaded", False):  
                context = st.session_state.get("file_text", "No file content available.")  
            else:  
                docs, urls = process_and_store_web_documents(query)  
                context = " ".join(docs)  
                st.subheader("📄 Scraped Content")  
                for url, doc in zip(urls, docs):  
                    with st.expander(url):  
                        st.write(doc[:1000] + "...")  
            
            st.subheader("🤖 AI Response")  
            response = query_ollama(query, context)  
            st.write(response)  
            generate_cloud(context)
            
            st.session_state.chat_history.append({"query": query, "response": response})  

if __name__ == "__main__":  
    main()
