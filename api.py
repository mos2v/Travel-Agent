from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from src.travel_plan_v3 import DataLoader, DocumentProcessor, VectorStoreManager, LLMService
from src.token_manager import TokenManager
from src.session_manager import SessionManager, create_travel_session
from src.packing_list import LLMPackingList, processPackingList
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
    session_manager = SessionManager(expiration_hours=2, persist_to_file=True)  # 2-hour sessions
    
    logger.info("🔁 Loading data...")
    data = DataLoader()

    logger.info("📄 Processing documents...")
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.load_or_process_documents()

    logger.info("🧠 Loading vector store...")
    vector_store = VectorStoreManager(documents)
    retriever = vector_store.get_retriever()

    logger.info("🎒 Loading packing list data...")
    packing_list_processor = processPackingList("Extended_Egypt_Packing_List_with_New_Items.csv")
    packing_documents = packing_list_processor.df_to_documents()
    packing_vector_store = VectorStoreManager(documents=packing_documents, path="packing_list")
    packing_list_retriever = packing_vector_store.get_retriever()

    logger.info("⚡ Initializing LLM...")
    
    model_name, provider = token_manager.get_current_model()
    temperature = token_manager.get_model_temperature()
    
    llm_manager = LLMService(model_name, provider=provider, temperature=temperature)
    
    logger.info(f"✅ App initialized with model: {model_name} (provider: {provider}, temperature: {temperature})")

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
    
    global llm_manager, token_manager, session_manager
    
    try:
        # Calculate number of days from start and end dates
        from datetime import datetime
        try:
            start_date_obj = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(request.end_date, "%Y-%m-%d")
            num_days = (end_date_obj - start_date_obj).days + 1  # +1 to include both start and end days
            
            if num_days <= 0:
                raise HTTPException(status_code=400, detail="End date must be after start date")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}")
        
        # Check if we need to switch models before processing
        current_model_name, current_provider = token_manager.get_current_model()
        
        # Get the default temperature for the current model
        default_temperature = token_manager.get_model_temperature()
        
        # Use request temperature if provided, otherwise use default
        temperature = request.temperature if request.temperature is not None else default_temperature
        
        # If the current model doesn't match the LLM manager's model or temperature changed, update it
        if (llm_manager.model_name != current_model_name or 
            llm_manager.provider != current_provider or
            llm_manager.temperature != temperature):
            logger.info(f"🔄 Switching model to {current_model_name} (provider: {current_provider}, temperature: {temperature})")
            llm_manager = LLMService(current_model_name, provider=current_provider, temperature=temperature)
        
        # Get the callback handler for token tracking
        callback_handler = token_manager.get_callback_handler()
        
        # Generate travel plan using calculated num_days
        travel_plan = llm_manager.travel_plan(
            retriever,
            request.city,
            request.favorite_places,
            request.visitor_type,
            str(num_days),  # Convert to string as expected by travel_plan
            request.budget,
            callbacks=[callback_handler]
        )
        
        # Update token usage after generation
        usage_stats = token_manager.update_usage_from_callback()
        logger.info(f"Request completed. Tokens used: {usage_stats['total_tokens']}")
        
        # Create session with travel plan and user parameters (including dates)
        user_params = {
            "city": request.city,
            "favorite_places": request.favorite_places,
            "visitor_type": request.visitor_type,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "num_days": num_days,  # Store calculated num_days
            "budget": request.budget
        }
        
        # Create session and return travel plan with session info
        session_response = create_travel_session(travel_plan, user_params, session_manager)
        
        # Add calculated num_days to the response for reference
        session_response["calculated_days"] = num_days
        
        return session_response
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")


@app.post("/api/packing-list")
async def generate_packing_list(request: PackingListRequest):
    """
    Generate a packing list based on a travel plan.
    Can use session_id from previous travel plan generation or accept travel plan directly.
    """
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
                
                # Extract city and dates from session if not provided in request
                if not city:
                    city = user_params.get("city")
                if not start_date:
                    start_date = user_params.get("start_date")
                if not end_date:
                    end_date = user_params.get("end_date")
                    
                logger.info(f"✅ Using travel plan from session {request.session_id}")
            else:
                logger.warning(f"⚠️ Session {request.session_id} not found or expired")
        
        # Fallback to provided travel plan
        if not travel_plan_data and request.travel_plan:
            travel_plan_data = request.travel_plan
            logger.info("✅ Using travel plan provided in request")
        
        # If we still don't have a travel plan, return error
        if not travel_plan_data:
            raise HTTPException(
                status_code=400, 
                detail="No travel plan available. Provide either a valid session_id or include travel_plan in the request."
            )
        
        # Validate that we have necessary information
        if not city:
            raise HTTPException(
                status_code=400,
                detail="City information is required. Provide city in request or use a valid session_id."
            )
        
        # Get current model settings
        current_model_name, current_provider = token_manager.get_current_model()
        default_temperature = token_manager.get_model_temperature()
        
        # Initialize packing list LLM
        packing_llm = LLMPackingList(
            model_name=current_model_name, 
            provider=current_provider, 
            temperature=default_temperature
        )
        
        # Generate packing list with enhanced context
        logger.info("🎒 Generating packing list...")
        packing_list_result = packing_llm.generate_packing_list(
            packing_list_retriever,
            travel_plan_data,
            city=city,
            start_date=start_date,
            end_date=end_date
        )
        
        # If we used a session, extend its expiration as a courtesy
        if request.session_id and session_data:
            session_manager.extend_session(request.session_id, hours=1)
        
        return {
            "packing_list": packing_list_result.content if hasattr(packing_list_result, 'content') else str(packing_list_result),
            "session_id": request.session_id,
            "based_on_session": bool(request.session_id and session_data),
            "city": city,
            "start_date": start_date,
            "end_date": end_date
        }
        
    except HTTPException:
        raise
    except Exception as e:
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

@app.post("/api/set-temperature")
async def set_model_temperature(model_name: str = Query(..., description="Model name to update temperature for"), 
                               temperature: float = Query(..., description="Temperature value (0.0-1.0)")):
    """Endpoint to set the temperature for a specific model."""
    global token_manager
    if token_manager is None:
        return {"error": "Token manager not initialized"}
    
    # Ensure temperature is within valid range
    temperature = max(0.0, min(1.0, temperature))
    
    success = token_manager.set_model_temperature(model_name, temperature)
    if success:
        return {"status": "success", "message": f"Temperature for model {model_name} set to {temperature}"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


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