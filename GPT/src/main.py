from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator # Note: validator is also deprecated/moved, see below
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from simple_processor import DocumentProcessor
import uvicorn
import os
import asyncio
import socket
import logging
import uuid
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import psutil

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    API_KEY: Optional[str] = None
    FRONTEND_URL: Optional[str] = None
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    RELEVANCE_THRESHOLD: float = 0.2
    MAX_CACHE_AGE: int = 3600  # 1 hour in seconds
    
    class Config:
        env_file = ".env"

# Load settings
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s'
)

# Add request ID to logging context
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True

logging.getLogger().addFilter(RequestIDFilter())

# Initialize document processor
try:
    processor = DocumentProcessor()
except Exception as e:
    logging.error(f"Error initializing document processor: {str(e)}")
    raise

class ErrorResponse(BaseModel):
    """Model for error responses."""
    detail: str
    status_code: int
    error_type: Optional[str] = None
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and process documents on startup."""
    try:
        # Use absolute path to the data file
        current_dir = os.path.dirname(os.path.abspath(_file_))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'scraped_docs.json')
        if not os.path.exists(data_path):
            logging.warning(f"Data file not found at {data_path}")
            return
        documents = processor.load_documents(data_path)
        logging.info(f"Loaded {len(documents)} documents")
        processor.process_documents(documents)
        logging.info("Documents processed successfully")
    except Exception as e:
        logging.error(f"Error during startup: {str(e)}", exc_info=True)
        raise
    yield
    try:
        # Cleanup
        if hasattr(processor, 'cleanup'):
            await processor.cleanup()
            logging.info("Document processor cleanup completed")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}", exc_info=True)

# Initialize API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key if provided."""
    if settings.ENVIRONMENT.lower() == "production":
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required in production"
            )
        if api_key != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    return api_key

app = FastAPI(
    title="Documentation Chatbot API",
    description="API for searching and retrieving documentation content",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Development
        "http://127.0.0.1:8000",  # Development
        os.getenv("FRONTEND_URL", "*")  # Production URL from environment
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Get the absolute path to the static directory
current_dir = os.path.dirname(os.path.abspath(_file_))
static_dir = os.path.join(current_dir, "static")

# Validate static directory exists
if not os.path.exists(static_dir):
    logging.error(f"Static directory not found at {static_dir}")
    raise RuntimeError(f"Static directory not found at {static_dir}")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logging.info(f"Static files mounted from {static_dir}")
except Exception as e:
    logging.error(f"Failed to mount static files: {str(e)}")
    raise

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to each request for tracking."""
    request_id = str(uuid.uuid4())
    logging.LoggerAdapter(logging.getLogger(), {'request_id': request_id})
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

def create_static_response(path: str, media_type: Optional[str] = None) -> FileResponse:
    """Create a FileResponse with proper caching headers."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
        
    headers = {
        "Cache-Control": f"public, max-age={settings.MAX_CACHE_AGE}",
        "Expires": (datetime.utcnow() + timedelta(seconds=settings.MAX_CACHE_AGE)).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
    }
    
    return FileResponse(
        path,
        media_type=media_type,
        headers=headers
    )

@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(static_dir, "img", "favicon.ico")
    return create_static_response(favicon_path, "image/x-icon")

@app.get("/")
async def root():
    """Serve the chatbot interface."""
    index_path = os.path.join(static_dir, "index.html")
    return create_static_response(index_path, "text/html")

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The search query text")
    max_results: Optional[int] = Field(3, ge=1, le=10, description="Maximum number of results to return")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Query text cannot be empty or just whitespace")
        
        # Remove any potentially harmful characters
        v = re.sub(r'[^\w\s\-\.,\?!\'"]', '', v)
        
        # Normalize whitespace
        v = ' '.join(v.split())
        
        return v

class SearchResponse(BaseModel):
    content: str = Field(..., description="The content of the found document")
    source: str = Field(..., description="Source of the document")
    title: str = Field(..., description="Title of the document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")

@app.post(
    "/search",
    response_model=List[SearchResponse],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        429: {"model": ErrorResponse, "description": "Too many requests"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    tags=["Search"]
)
async def search(
    query: Query,
    api_key: str = Depends(verify_api_key)
):
    """
    Search for relevant documentation based on query.

    Parameters:
    - text: Search query text (1-1000 characters)
    - max_results: Maximum number of results to return (1-10, default: 3)

    Returns:
    - List of matching documents with relevance scores

    Note on rate limiting:
    - Consider implementing rate limiting for production use
    - Recommended: max 10 requests per minute per client
    """
    try:
        # Log the search request
        logging.info(f"Search request received: {query.text[:50]}...")
        
        # Get search results
        results = processor.search_documents(query.text, k=query.max_results)
        
        # Convert to response objects
        filtered_results = []
        for result in results:
            if result['relevance_score'] >= 0.2:  # Must be relevant enough
                filtered_results.append(
                    SearchResponse(
                        content=result['content'],
                        source=result['metadata']['source'],
                        title=result['metadata']['title'],
                        relevance_score=result['relevance_score']
                    )
                )
        
        # Log the number of results found
        logging.info(f"Found {len(filtered_results)} relevant results")
        
        # If no good matches, provide a helpful message
        if not filtered_results:
            return [
                SearchResponse(
                    content="I couldn't find a specific answer to your question. Please try rephrasing your question or ask something else.",
                    source="system",
                    title="No Match Found",
                    relevance_score=0.0
                )
            ]
            
        return filtered_results
    except Exception as e:
        logging.error("Error in search endpoint", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                detail=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_type="SEARCH_ERROR"
            ).dict()
        )

@app.get("/documents/status", tags=["Documents"])
async def document_status():
    """
    Get the current status of loaded documents.
    
    Returns:
    - total_documents: Total number of documents loaded
    - document_sources: List of unique document sources
    - last_updated: Timestamp of last document update
    """
    try:
        doc_count = len(processor.documents) if processor.documents else 0
        sources = list(set(doc['metadata']['source'] for doc in processor.documents)) if processor.documents else []
        
        return {
            "total_documents": doc_count,
            "document_sources": sources,
            "last_updated": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logging.error("Failed to get document status", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                detail="Failed to get document status",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_type="DOCUMENT_STATUS_ERROR"
            ).dict()
        )

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns:
    - status: Current system status
    - documents_loaded: Number of documents loaded
    - timestamp: Current server time
    - environment: Current environment
    """
    try:
        doc_count = len(processor.documents) if processor.documents else 0
        memory_info = psutil.Process().memory_info()
        return {
            "status": "healthy",
            "documents_loaded": doc_count,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "system_info": {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
        }
    except Exception as e:
        logging.error("Health check failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                detail="System health check failed",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_type="HEALTH_CHECK_ERROR"
            ).dict()
        )

def find_available_port(start_port=8000, max_port=8020):
    """Find an available port between start_port and max_port."""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports in range {start_port}-{max_port}")

if _name_ == "_main_":
    try:
        # Find available port
        port = find_available_port()
        logging.info(f"Starting server on port {port}")
        
        # Configure server
        config = uvicorn.Config(
            "main:app",
            host="127.0.0.1",  # Using localhost for development
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
        
        # Start server
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except Exception as e:
        logging.error("Failed to start server", exc_info=True)
        raise