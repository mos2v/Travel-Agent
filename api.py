from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from src.travel_plan_v3 import DataLoader, DocumentProcessor, VectorStoreManager, LLMService
from src.token_manager import TokenManager
from src.session_manager import SessionManager, create_travel_session
from src.packing_list import PackingListGenerator, PackingListProcessor
from datetime import datetime
import traceback
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TravelAPI")

# Initialize token manager
token_manager = None
session_manager = None
packing_list_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_manager, token_manager, session_manager, packing_list_retriever
    load_dotenv()
    
    token_manager = TokenManager()
    session_manager = SessionManager(expiration_hours=2, persist_to_file=True)
    
    logger.info("🔁 Loading data...")
    data = DataLoader()

    logger.info("📄 Processing documents...")
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.load_or_process_documents()

    logger.info("🧠 Creating shared embedding model (E5-Large)...")
    # Create a single shared embedding model instance to optimize memory usage
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    
    shared_embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("✅ Shared embedding model loaded successfully")

    logger.info("🧠 Loading travel planning vector store...")
    vector_store = VectorStoreManager(documents, embeddings_instance=shared_embeddings)
    retriever = vector_store.get_retriever()

    logger.info("🎒 Loading packing list data...")
    packing_list_processor = PackingListProcessor("Extended_Egypt_Packing_List_with_New_Items.csv")
    packing_documents = packing_list_processor.df_to_documents()
    
    logger.info("🎒 Loading packing list vector store (reusing embedding model)...")
    packing_vector_store = VectorStoreManager(
        documents=packing_documents, 
        embeddings_instance=shared_embeddings,  # Reuse the same embedding model instance
        path="packing_list"
    )
    packing_list_retriever = packing_vector_store.get_retriever()

    logger.info("⚡ Initializing LLM...")
    model_name, provider = token_manager.get_current_model()
    temperature = token_manager.get_model_temperature()
    llm_manager = LLMService(model_name, provider=provider, temperature=temperature)
    
    logger.info(f"✅ App initialized with model: {model_name} (provider: {provider}, temperature: {temperature})")
    logger.info("🚀 Memory optimized: Using single E5-Large instance for both vector stores")

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
session_manager = None
packing_list_retriever = None



class TravelPlanRequest(BaseModel):
    city: str
    favorite_places: str
    visitor_type: str
    start_date: str = Field(description="Trip start date (YYYY-MM-DD format)")
    end_date: str = Field(description="Trip end date (YYYY-MM-DD format)")
    budget: str
    temperature: float = Field(default=None, description="Optional temperature parameter for model generation")


class PackingListRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID from travel plan generation")
    travel_plan: Optional[dict] = Field(default=None, description="Full travel plan object (fallback if session expired)")
    city: Optional[str] = Field(default=None, description="City for the trip (will be extracted from session if not provided)")
    start_date: Optional[str] = Field(default=None, description="Trip start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="Trip end date (YYYY-MM-DD)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "travel_session_abc123",
                "start_date": "2025-07-01",
                "end_date": "2025-07-03"
            }
        }

@app.post("/api/travel-plan")
async def generate_travel_plan(request: TravelPlanRequest):
    """Generate a travel plan based on user requirements."""
    global llm_manager, token_manager, session_manager
    
    try:
        # Calculate number of days
        start_date_obj = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(request.end_date, "%Y-%m-%d")
        num_days = (end_date_obj - start_date_obj).days + 1
        
        if num_days <= 0:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        # Get current model configuration
        current_model_name, current_provider = token_manager.get_current_model()
        default_temperature = token_manager.get_model_temperature()
        temperature = request.temperature if request.temperature is not None else default_temperature
        
        # Update LLM manager if needed
        if (llm_manager.model_name != current_model_name or 
            llm_manager.provider != current_provider or
            llm_manager.temperature != temperature):
            logger.info(f"🔄 Switching model to {current_model_name} (provider: {current_provider}, temperature: {temperature})")
            llm_manager = LLMService(current_model_name, provider=current_provider, temperature=temperature)
        
        # Generate travel plan
        callback_handler = token_manager.get_callback_handler()
        travel_plan = llm_manager.travel_plan(
            retriever, request.city, request.favorite_places, request.visitor_type,
            str(num_days), request.budget, callbacks=[callback_handler]
        )
        
        # Update token usage
        usage_stats = token_manager.update_usage_from_callback()
        logger.info(f"Request completed. Tokens used: {usage_stats['total_tokens']}")
        
        # Create session
        user_params = {
            "city": request.city, "favorite_places": request.favorite_places,
            "visitor_type": request.visitor_type, "start_date": request.start_date,
            "end_date": request.end_date, "num_days": num_days, "budget": request.budget
        }
        
        session_response = create_travel_session(travel_plan, user_params, session_manager)
        session_response["calculated_days"] = num_days
        
        return session_response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating travel plan: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")


@app.post("/api/packing-list")
async def generate_packing_list(request: PackingListRequest):
    """Generate a packing list based on a travel plan."""
    global session_manager, packing_list_retriever, token_manager
    
    try:
        travel_plan_data = None
        user_params = None
        city = request.city
        start_date = request.start_date
        end_date = request.end_date
        
        # Try to get travel plan from session first
        if request.session_id:
            session_data = session_manager.get_session(request.session_id)
            if session_data:
                travel_plan_data = session_data["travel_plan"]
                user_params = session_data["user_params"]
                
                # Extract missing info from session
                city = city or user_params.get("city")
                start_date = start_date or user_params.get("start_date")
                end_date = end_date or user_params.get("end_date")
                
                logger.info(f"✅ Using travel plan from session {request.session_id}")
            else:
                logger.warning(f"⚠️ Session {request.session_id} not found or expired")
        
        # Fallback to provided travel plan
        if not travel_plan_data and request.travel_plan:
            travel_plan_data = request.travel_plan
            logger.info("✅ Using travel plan provided in request")
        
        # Validate requirements
        if not travel_plan_data:
            raise HTTPException(status_code=400, 
                detail="No travel plan available. Provide either a valid session_id or include travel_plan in the request.")
        
        if not city:
            raise HTTPException(status_code=400,
                detail="City information is required. Provide city in request or use a valid session_id.")
        
        # Get current model settings and generate packing list
        current_model_name, current_provider = token_manager.get_current_model()
        default_temperature = token_manager.get_model_temperature()
        
        packing_llm = PackingListGenerator(
            model_name=current_model_name, 
            provider=current_provider, 
            temperature=default_temperature
        )
        
        logger.info("🎒 Generating packing list...")
        logger.info(f"🔧 Using model: {current_model_name} (provider: {current_provider}, temp: {default_temperature})")
        
        # Generate packing list with token tracking
        callback_handler = token_manager.get_callback_handler()
        packing_list_result = packing_llm.generate_packing_list(
            packing_list_retriever, travel_plan_data,
            city=city, start_date=start_date, end_date=end_date
        )
        
        # Update token usage
        usage_stats = token_manager.update_usage_from_callback()
        logger.info(f"Packing list generation completed. Tokens used: {usage_stats['total_tokens']}")
        
        # Extend session if used
        if request.session_id and session_data:
            session_manager.extend_session(request.session_id, hours=1)
        
        return {
            "packing_list": packing_list_result,
            "session_id": request.session_id,
            "based_on_session": bool(request.session_id and session_data),
            "city": city,
            "start_date": start_date,
            "end_date": end_date
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in packing list generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating packing list: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a travel plan session without returning the full data.
    Useful for checking if a session is still valid.
    """
    global session_manager
    
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    session_info = session_manager.get_session_info(session_id)
    
    if not session_info["exists"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_info["is_expired"]:
        raise HTTPException(status_code=410, detail="Session has expired")
    
    return session_info


@app.post("/api/session/{session_id}/extend")
async def extend_session(session_id: str, hours: int = Query(default=1, description="Hours to extend the session")):
    """
    Extend the expiration time of a session.
    """
    global session_manager
    
    if not session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    
    success = session_manager.extend_session(session_id, hours)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": f"Session extended by {hours} hour(s)", "session_id": session_id}

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

@app.post("/api/set-model")
async def set_model_configuration(
    model_name: str = Query(..., description="Model name to use"), 
    provider: str = Query(..., description="Provider (groq, nvidia, mistralai, google-genai)"),
    temperature: float = Query(default=None, description="Temperature value (0.0-1.0), optional")
):
    """Endpoint to set the model, provider, and optionally temperature."""
    global token_manager, llm_manager
    if token_manager is None:
        return {"error": "Token manager not initialized"}
    
    # Validate provider
    valid_providers = ['groq', 'nvidia', 'mistralai', 'google-genai']
    if provider not in valid_providers:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}"
        )
    
    # Set the model and provider in token manager
    success = token_manager.set_current_model(model_name, provider)
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to set model {model_name} with provider {provider}"
        )
    
    # Set temperature if provided
    if temperature is not None:
        # Ensure temperature is within valid range
        temperature = max(0.0, min(1.0, temperature))
        temp_success = token_manager.set_model_temperature(model_name, temperature)
        if not temp_success:
            # Model was set but temperature failed - still return success for model change
            logger.warning(f"Model set successfully but failed to set temperature for {model_name}")
    
    # Update the LLM manager with new configuration
    try:
        current_temperature = token_manager.get_model_temperature()
        llm_manager = LLMService(model_name, provider=provider, temperature=current_temperature)
        logger.info(f"🔄 Updated LLM manager: {model_name} (provider: {provider}, temperature: {current_temperature})")
    except Exception as e:
        logger.error(f"Failed to update LLM manager: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Model and provider set but failed to initialize LLM: {str(e)}"
        )
    
    response_data = {
        "status": "success", 
        "model_name": model_name,
        "provider": provider,
        "temperature": token_manager.get_model_temperature(),
        "message": f"Model set to {model_name} with provider {provider}"
    }
    
    return response_data


@app.get("/api/session-stats")
async def get_session_stats():
    """Endpoint to check current session statistics."""
    global session_manager
    if session_manager is None:
        return {"error": "Session manager not initialized"}
    
    return {
        "active_sessions": session_manager.get_active_sessions_count(),
        "expiration_hours": session_manager.expiration_hours,
        "persist_to_file": session_manager.persist_to_file
    }

@app.get("/api/model-config")
async def get_model_configuration():
    """Endpoint to get the current model configuration."""
    global token_manager
    if token_manager is None:
        return {"error": "Token manager not initialized"}
    
    try:
        model_name, provider = token_manager.get_current_model()
        temperature = token_manager.get_model_temperature()
        
        return {
            "status": "success",
            "model_name": model_name,
            "provider": provider,
            "temperature": temperature,
            "available_providers": ['groq', 'nvidia', 'mistralai', 'google-genai']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model configuration: {str(e)}")

@app.get("/api/available-models")
async def get_available_models():
    """Get available models for each provider."""
    available_models = {
        # "groq": [
        #     "meta-llama/llama-4-maverick-17b-128e-instruct",
        #     "meta-llama/llama-4-scout-17b-16e-instruct"
        # ],
        "nvidia": [
            "meta/llama-3.3-70b-instruct"
        ],
        "mistralai": [
            "mistral-large-latest",
            "magistral-medium-2506",
        ],
        "google-genai": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]
    }
    
    return {
        "status": "success",
        "available_models": available_models,
        "total_providers": len(available_models),
        "total_models": sum(len(models) for models in available_models.values())
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Travel Planner API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🔄 Use Ctrl+C to stop the server")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )