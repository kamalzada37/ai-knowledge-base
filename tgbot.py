import re
import io
import os
import requests
import chromadb
import fitz  # PyMuPDF for PDF processing
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
import ollama
from langchain_ollama import OllamaEmbeddings
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Constants
TELEGRAM_BOT_TOKEN = "7929207317:AAHEmS2wQ6msVMx7ZujXqfAK8kzqhvAyUhw"
LLM_MODEL = "llama3.2"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "knowledge_base"
ADMIN_CHAT_ID = 6529120305  # Replace with your actual admin chat id

# Global chat history stored per user
chat_history = {}

# --- Embedding and DB Initialization ---
class EmbeddingWrapper:
    def __init__(self, embed_func, dimension):
        self.embed_func = embed_func
        self.dimension = dimension
    def __call__(self, input):
        embeddings = self.embed_func.embed_documents(input)
        if len(embeddings[0]) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embeddings[0])} does not match expected {self.dimension}")
        return embeddings

embedding_function = OllamaEmbeddings(model=LLM_MODEL, base_url="http://localhost:11434")
embed_wrapper = EmbeddingWrapper(embedding_function, dimension=3072)

def initialize_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_wrapper,
        metadata={"dimension": 3072}
    )
    return collection

collection = initialize_chromadb()

def sanitize_text(text):
    """Replace predefined swear words with asterisks."""
    swear_words = ["badword1", "badword2", "badword3"]
    pattern = re.compile(r'\b(' + '|'.join(swear_words) + r')\b', re.IGNORECASE)
    return pattern.sub("****", text)

def query_ollama(query: str) -> str:
    sanitized_query = sanitize_text(query)
    results = collection.query(query_texts=[sanitized_query], n_results=5)
    context_text = ""
    if results.get('documents') and results['documents'][0]:
        context_text = "\n\n".join(results['documents'][0])
    else:
        context_text = "No relevant documents found."
    prompt = [{"role": "user", "content": f"Context: {context_text}\n\nQuestion: {sanitized_query}\nAnswer:"}]
    try:
        response = ollama.chat(model=LLM_MODEL, messages=prompt)
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

def advanced_query_processing(query: str) -> str:
    """
    Placeholder for advanced multi-step reasoning or agent-based processing.
    Currently, it simply calls query_ollama.
    """
    return query_ollama(query)

def send_callback_notification(query: str, response: str, context: CallbackContext):
    """
    Sends a callback notification to the admin if the query meets specific criteria.
    For example, if the query contains the word 'urgent'.
    """
    if "urgent" in query.lower():
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Callback Notification:\nQuery: {query}\nResponse: {response}")

def generate_wordcloud_text() -> str:
    try:
        all_docs = collection.get()  # returns dict with 'documents'
        if 'documents' in all_docs:
            combined = "\n\n".join(all_docs['documents'])
            return combined
    except Exception:
        results = collection.query(query_texts=[""], n_results=1000)
        if results.get('documents') and results['documents'][0]:
            return "\n\n".join(results['documents'][0])
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

async def start(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    chat_history.setdefault(user_id, [])
    welcome = (
        "Hello! I am your AI-Powered Knowledge Base Bot.\n\n"
        "You can:\n"
        "• Upload a document (PDF or text file) to add to the knowledge base.\n"
        "• Send a web link to add that webpage's content.\n"
        "• Ask a question to get an answer based on the stored knowledge.\n"
        "• Use /wordcloud to view an insight visualization of the knowledge base.\n"
        "• Use /history to see your chat history.\n"
    )
    await update.message.reply_text(welcome)

async def handle_document(update: Update, context: CallbackContext) -> None:
    document = update.message.document
    filename = document.file_name
    file = await document.get_file()
    file_bytes = await file.download_as_bytearray()
    process_document(file_bytes, filename)
    await update.message.reply_text(f"Document '{filename}' has been processed and added to the knowledge base.")

def process_document(file_bytes: bytes, filename: str) -> None:
    if filename.lower().endswith('.pdf'):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        collection.add(documents=[text], metadatas=[{"source": filename}])
    elif filename.lower().endswith('.txt'):
        text = file_bytes.decode('utf-8')
        collection.add(documents=[text], metadatas=[{"source": filename}])
    else:
        raise ValueError("Unsupported file type")

def process_web_document(url: str) -> bool:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        collection.add(documents=[text], metadatas=[{"source": url}])
        return True
    except Exception as e:
        print(f"Failed to process web document: {e}")
        return False

async def handle_text(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    text_msg = update.message.text.strip()
    if re.match(r'^https?:\/\/', text_msg):
        processed = process_web_document(text_msg)
        if processed:
            await update.message.reply_text(f"Web page '{text_msg}' has been processed and added to the knowledge base.")
        else:
            await update.message.reply_text(f"Failed to process the web page '{text_msg}'.")
    else:
        answer = advanced_query_processing(text_msg)
        chat_history.setdefault(user_id, []).append({"query": text_msg, "response": answer})
        await update.message.reply_text(answer)
        send_callback_notification(text_msg, answer, context)

async def wordcloud_handler(update: Update, context: CallbackContext) -> None:
    combined_text = generate_wordcloud_text()
    if combined_text:
        img_buf = generate_wordcloud_image(combined_text)
        await update.message.reply_photo(photo=img_buf, caption="Word Cloud of the Knowledge Base")
    else:
        await update.message.reply_text("No content available for word cloud generation.")

async def history_handler(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    history = chat_history.get(user_id, [])
    if history:
        message = ""
        for entry in history:
            message += f"Q: {entry['query']}\nA: {entry['response']}\n\n"
        await update.message.reply_text(message)
    else:
        await update.message.reply_text("No chat history found.")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("wordcloud", wordcloud_handler))
    application.add_handler(CommandHandler("history", history_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.run_polling()

if __name__ == "__main__":
    main()
