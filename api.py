from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from src.travel_plan_v2 import DataLoader, DocumentProcessor, VectorStoreManager, LLMService
from src.token_manager import TokenManager
import traceback
import os 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TravelAPI")

# Initialize token manager
token_manager = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_manager, token_manager
    load_dotenv()
    
    token_manager = TokenManager()
    
    logger.info("🔁 Loading data...")
    data = DataLoader()

    logger.info("📄 Processing documents...")
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.load_or_process_documents()

    logger.info("🧠 Loading vector store...")
    vector_store = VectorStoreManager(documents)
    retriever = vector_store.get_retriever()

    logger.info("⚡ Initializing LLM...")
    
    model_name, provider = token_manager.get_current_model()
    
    llm_manager = LLMService(model_name, provider=provider)
    
    logger.info(f"✅ App initialized with model: {model_name} (provider: {provider})")

    yield
    logger.info("🔻 Shutting down...")



app = FastAPI(title="Travel Planner API", lifespan=lifespan)

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
llm_manager = None



class TravelPlanRequest(BaseModel):
    city: str
    favorite_places: str
    visitor_type: str
    num_days: str
    budget: str

@app.post("/api/travel-plan")
async def generate_travel_plan(request: TravelPlanRequest):
    
    global llm_manager, token_manager
    
    try:
        # Check if we need to switch models before processing
        current_model_name, current_provider = token_manager.get_current_model()
        
        # If the current model doesn't match the LLM manager's model, update it
        if (llm_manager.model_name != current_model_name or 
            llm_manager.provider != current_provider):
            logger.info(f"🔄 Switching model to {current_model_name} (provider: {current_provider})")
            llm_manager = LLMService(current_model_name, provider=current_provider)
        
        # Get the callback handler for token tracking
        callback_handler = token_manager.get_callback_handler()
        
        
        travel_plan = llm_manager.travel_plan(
            retriever,
            request.city,
            request.favorite_places,
            request.visitor_type,
            request.num_days,
            request.budget,
            callbacks=[callback_handler]
        )
        
        # Update token usage after generation
        usage_stats = token_manager.update_usage_from_callback()
        logger.info(f"Request completed. Tokens used: {usage_stats['total_tokens']}")
        
        
        # Return the travel plan as JSON
        return travel_plan
    except Exception as e:
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/usage-stats")
async def get_usage_stats():
    """Endpoint to check current token usage statistics."""
    global token_manager
    if token_manager is None:
        return {"error": "Token manager not initialized"}
    
    return token_manager.get_usage_stats()