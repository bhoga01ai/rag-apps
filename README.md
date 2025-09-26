# RAG Applications

This repository contains a collection of Retrieval-Augmented Generation (RAG) applications demonstrating various vector databases and web scraping techniques.

## Features

*   **Semantic Search:** Perform semantic search using different vector databases:
    *   Qdrant
    *   Faiss
    *   pgvector
*   **Web Scraping:** Scrape content from URLs to build a knowledge base for the RAG applications.
*   **Streamlit User Interface:** A user-friendly web interface built with Streamlit for interacting with the Q&A system.
*   **LLM Integration:** Integrated with Large Language Models (LLMs) from OpenAI and Google to provide answers based on the retrieved context.

## Getting Started

### Prerequisites

*   Python 3.7+
*   An OpenAI API key
*   A Qdrant Cloud account and API key (for Qdrant examples)
*   Access to a PostgreSQL database (for the pgvector example)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bhoga01ai/rag-apps.git
    cd rag-apps
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add the following:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    QDRANT_API_KEY="your_qdrant_api_key"
    # Add other API keys or connection strings as needed
    ```

## Usage

This repository contains several applications. Here's how to run them:

### 1. Qdrant Streamlit App

This application provides a web interface to create and manage collections in a Qdrant database, upload documents, and ask questions.

**To run the FastAPI backend:**
```bash
uvicorn qdrant_api:app --reload
```

**To run the Streamlit frontend:**
```bash
streamlit run qdrant_streamlit_app.py
```

### 2. Web Scraping with FAISS

This Streamlit application scrapes content from URLs, creates a FAISS index, and allows you to ask questions based on the scraped content.

```bash
streamlit run web_scrape_app_faiss.py
```

### 3. Web Scraping with Qdrant

This Streamlit application scrapes content from URLs, creates a Qdrant collection, and allows you to ask questions based on the scraped content.

```bash
streamlit run web_scrape_app_qdrant.py
```

### 4. Similarity Search with pgvector

This script demonstrates how to perform similarity search using pgvector with a PostgreSQL database.

```bash
python similarity_search_pgvector.py
```

### 5. Qdrant Vector Database Example

This script shows a complete workflow of using Qdrant for semantic search, from data preparation to querying.

```bash
python llmops_vectordatabase_qdrant_01.py
```

## File Descriptions

*   `qdrant_api.py`: A FastAPI application that provides an API for interacting with the Qdrant vector database.
*   `qdrant_streamlit_app.py`: A Streamlit application that provides a UI for the Qdrant-based Q&A system.
*   `web_scrape_app_faiss.py`: A Streamlit application that demonstrates web scraping and building a Q&A system with FAISS.
*   `web_scrape_app_qdrant.py`: A Streamlit application that demonstrates web scraping and building a Q&A system with Qdrant.
*   `similarity_search_pgvector.py`: An example script for performing similarity search with pgvector.
*   `llmops_vectordatabase_qdrant_01.py`: A script that demonstrates the end-to-end process of using Qdrant for semantic search.
*   `main.py`: The main entry point of the application.
*   `requirements.txt`: A list of the Python dependencies for the project.
*   `sotu_address_obama.txt`: A text file containing a State of the Union address by Barack Obama, used as a sample document.
*   `faiss_index/`: A directory where the FAISS index is stored.
*   `.env`: A file for storing environment variables (e.g., API keys).
*   `.gitignore`: A file that specifies which files and directories to ignore in version control.
