from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from src.travel_plan_v1 import DataLoader, DocumentProcessor, VectorStoreManager, LLMService
import traceback





@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_manager
    load_dotenv()
    print("🔁 Loading data...")
    data = DataLoader()

    print("📄 Processing documents...")
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.load_or_process_documents()

    print("🧠 Loading vector store...")
    vector_store = VectorStoreManager(documents)
    retriever = vector_store.get_retriever()

    print("⚡ Initializing LLM...")
    llm_manager = LLMService("llama-3.3-70b-versatile")

    print("✅ App initialization complete!")

    yield
    print("🔻 Shutting down...")



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
    try:
        travel_plan = llm_manager.travel_plan(
            retriever,
            request.city,
            request.favorite_places,
            request.visitor_type,
            request.num_days,
            request.budget
        )
        
        # Return the travel plan as JSON
        return travel_plan
    except Exception as e:
        traceback.print_exc()  # Add this line
        raise HTTPException(status_code=500, detail=f"Error generating travel plan: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

