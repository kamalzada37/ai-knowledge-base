# AI-Powered Collaborative Knowledge Base

This repository contains an AI-powered knowledge base application built using Streamlit, LangChain, Ollama, and Chroma. The project enables users to contribute documents (PDFs, text files, or web links) and interact with the stored content using natural language queries. **All functionalities are available via a Telegram Bot interface** as required.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Telegram Bot](#telegram-bot)
  - [(Optional) Web Interface](#optional-web-interface)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Team and Contribution](#team-and-contribution)
- [License](#license)

## Project Overview
This project is a collaborative knowledge base that allows users to:
- **Contribute Documents:** Upload PDFs, text files, or submit web links.
- **Query the Knowledge Base:** Ask questions or request summaries of the contributed documents using a Retrieval-Augmented Generation (RAG) pipeline.
- **Visualize Insights:** Generate word clouds to visualize trends and topics across the stored content.
- **Interact via Telegram:** The entire functionality is available through a Telegram bot, making it accessible on-the-go.

## Features
- **Document Contribution System:** Users can upload files or submit URLs; documents are processed, chunked, and stored in a Chroma vector database.
- **Natural Language Querying:** Uses Ollama with a RAG approach to provide contextually relevant responses.
- **Insight Visualization:** Generate word clouds from aggregated document data.
- **Telegram Bot Integration:** All features are accessible through Telegram commands and file uploads.
- **Chat History:** Maintains a per-user chat history for reference.

## Tech Stack
- **Backend:** Python, LangChain, Ollama, ChromaDB
- **Web Interface:** Streamlit (optional, if needed for demo)
- **Bot Interface:** Telegram Bot API
- **Data Processing:** PyMuPDF (for PDFs), BeautifulSoup (for web scraping), RecursiveCharacterTextSplitter
- **Visualization:** Matplotlib, WordCloud

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/@kamalzada37/ai-knowledge-base.git
   cd ai-knowledge-base


(Optional) Web Interface
If needed for demos or internal testing, you can also run the Streamlit app:

bash
streamlit run app.py
Note: The primary interaction is via Telegram as per project requirements.

Project Structure
ai-knowledge-base/
├── app.py             # Streamlit web interface (optional demo)
├── tgbot.py           # Telegram bot interface
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── chroma_db/         # Chroma persistent database directory (auto-created)
