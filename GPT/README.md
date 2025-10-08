# Documentation Chatbot

A sophisticated chatbot that scrapes documentation content and provides intelligent responses using modern NLP techniques.

## Features

- Documentation scraping with BeautifulSoup4
- Vector embeddings using Sentence Transformers
- Efficient vector search with ChromaDB
- FastAPI backend for quick responses
- Automatic content chunking and processing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the scraper:
   - Edit the URLs in `src/scraper.py`
   - Run the scraper: `python src/scraper.py`

3. Start the API:
```bash
python src/main.py
```

## Usage

The API will be available at `http://localhost:8000`

### Endpoints

- `POST /search`: Search documentation
  ```json
  {
    "text": "How do I create a loyalty program?",
    "max_results": 3
  }
  ```

- `GET /health`: Health check endpoint

## Project Structure

```
.
├── src/
│   ├── scraper.py      # Documentation scraping
│   ├── processor.py    # Document processing and vector search
│   └── main.py         # FastAPI application
├── data/               # Scraped data and vector store
└── requirements.txt
```
