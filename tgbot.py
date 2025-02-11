import re  
import io  
import fitz  # PyMuPDF for PDF processing  
import requests  
import chromadb  
import matplotlib.pyplot as plt  
from bs4 import BeautifulSoup  
from wordcloud import WordCloud  
from telegram import Update  
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext  
from langchain_ollama import OllamaEmbeddings  
import ollama  

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"    
CHROMA_DB_PATH = "chroma_db"  
COLLECTION_NAME = "knowledge_base"  
ADMIN_CHAT_ID = 123456789  
LLM_MODEL = "llama3.2"  

chat_history = {}  
user_data = {}  

def initialize_chromadb():  
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)  
    return client.get_or_create_collection(name=COLLECTION_NAME)  

collection = initialize_chromadb()  

def sanitize_text(text):  
    swear_words = ["badword1", "badword2", "badword3"]  
    pattern = re.compile(r'\b(' + '|'.join(swear_words) + r')\b', re.IGNORECASE)  
    return pattern.sub("****", text)  

def process_document(file_bytes: bytes, filename: str) -> str:  
    if filename.lower().endswith('.pdf'):  
        doc = fitz.open(stream=file_bytes, filetype="pdf")  
        text = ""  
        for page in doc:  
            text += page.get_text()  
    elif filename.lower().endswith('.txt'):  
        text = file_bytes.decode('utf-8')  
    else:  
        raise ValueError("Unsupported file type")  
    
    sanitized_text = sanitize_text(text)  
    collection.add(documents=[sanitized_text], metadatas=[{"source": filename}])  
    return sanitized_text  

def process_web_document(url: str) -> bool:  
    try:  
        response = requests.get(url)  
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')  
        text = soup.get_text()  
        sanitized_text = sanitize_text(text)  
        collection.add(documents=[sanitized_text], metadatas=[{"source": url}])  
        return True  
    except Exception as e:  
        print(f"Failed to process web document: {e}")  
        return False  

embedding_function = OllamaEmbeddings(model=LLM_MODEL, base_url="http://localhost:11434")  
def query_ollama(query: str) -> str:  
    sanitized_query = sanitize_text(query)  
    results = collection.query(query_texts=[sanitized_query], n_results=5)  
    context_text = "\n\n".join(results['documents'][0]) if results.get('documents') else "No relevant documents found."  
    
    prompt = [{"role": "user", "content": f"Context: {context_text}\n\nQuestion: {sanitized_query}\nAnswer:"}]  
    try:  
        response = ollama.chat(model=LLM_MODEL, messages=prompt)  
        return response['message']['content']  
    except Exception as e:  
        return f"Error: {e}"  

def generate_wordcloud_text() -> str:  
    try:  
        all_docs = collection.get()  
        return "\n\n".join(all_docs['documents']) if 'documents' in all_docs else ""  
    except Exception:  
        return ""  

def generate_wordcloud_image(text: str) -> io.BytesIO:  
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)  
    fig, ax = plt.subplots()  
    ax.imshow(wordcloud, interpolation='bilinear')  
    ax.axis("off")  
    buf = io.BytesIO()  
    plt.savefig(buf, format='png')  
    buf.seek(0)  
    plt.close(fig)  
    return buf  

async def handle_document(update: Update, context: CallbackContext) -> None:  
    user_id = update.effective_user.id  
    document = update.message.document  
    filename = document.file_name  
    file = await document.get_file()  
    file_bytes = await file.download_as_bytearray()  
    user_data[user_id] = {"file_text": process_document(file_bytes, filename)}  
    await update.message.reply_text(f"Document '{filename}' has been processed and added to the knowledge base.")  

async def handle_text(update: Update, context: CallbackContext) -> None:  
    user_id = update.effective_user.id  
    text_msg = update.message.text.strip()  
    
    if re.match(r'^https?:\/\/', text_msg):  
        processed = process_web_document(text_msg)  
        await update.message.reply_text(f"Web page '{text_msg}' has been {'processed' if processed else 'failed to process'}.")  
    else:  
        answer = query_ollama(text_msg)  
        chat_history.setdefault(user_id, []).append({"query": text_msg, "response": answer})  
        await update.message.reply_text(answer)  
        if "urgent" in text_msg.lower():  
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Urgent query from user: {text_msg}\nResponse: {answer}")  

async def wordcloud_handler(update: Update, context: CallbackContext) -> None:  
    combined_text = generate_wordcloud_text()  
    if combined_text:  
        img_buf = generate_wordcloud_image(combined_text)  
        await update.message.reply_photo(photo=img_buf, caption="Word Cloud of the Knowledge Base")  
    else:  
        await update.message.reply_text("No content available for word cloud generation.")  

async def start(update: Update, context: CallbackContext) -> None:  
    welcome = ("Hello! I am your AI-Powered Knowledge Base Bot.\n\n"  
               "Upload documents, ask questions, or analyze stored knowledge!")  
    await update.message.reply_text(welcome)  

def main():  
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()  
    application.add_handler(CommandHandler("start", start))  
    application.add_handler(CommandHandler("wordcloud", wordcloud_handler))  
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))  
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))  
    application.run_polling()  

if __name__ == "__main__":  
    main()
