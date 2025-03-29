import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import List, Optional
from pydantic import BaseModel, Field
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Rag_travel_planner_v1.0"

landmark_prices = pd.read_csv('data/egypt_v0.1.csv')
places_api_data = pd.read_csv('data/places_details.csv')

documents = []

for _, row in landmark_prices.iterrows():
    text = f"""
    Governorate: {row.get('Governorate/City', 'N/A')}
    Site: {row.get('Place', 'N/A')}
    Egyptian Ticket: {row.get('Egyptian', 'N/A')} EGP
    Egyptian Student Ticket: {row.get('EgyptianStudent', 'N/A')} EGP
    Foreign Ticket: {row.get('Foreign', 'N/A')} EGP
    Foreign Student Ticket: {row.get('ForeignStudent', 'N/A')} EGP
    Visiting Times: {row.get('VisitingTimes', 'N/A')}
    """
    documents.append(Document(page_content=text, metadata={"source": 'landmark_prices'}))

for _, row in places_api_data.iterrows():
    text = f"""
    Place Name: {row.get('displayName.text', 'N/A')}
    Place Primary Type: {row.get('primaryTypeDisplayName.text', 'N/A')}
    Place Types: {row.get('types', 'N/A')}
    Place Price: {row.get('priceRange.endPrice.units', 'N/A')} EGP
    Place Price Level: {row.get('priceLevel', 'N/A')}
    Place Location: {row.get('formattedAddress', 'N/A')}
    Place Star Rating: {row.get('rating', 'N/A')}
    Place website: {row.get('websiteUri', 'N/A')}
    """
    documents.append(Document(page_content=text, metadata={"source": 'Places_api', 'Type': f'{row["primaryTypeDisplayName.text"]}', 'city': f'{row["formattedAddress"]}'}))
    
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
path = Path('faiss_mpnetv2_v1.0')

if not path.exists():
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local('faiss_mpnetv2_v1.0')
else:
    vectorstore = FAISS.load_local('faiss_mpnetv2_v1.0', embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# llm_model = ChatNVIDIA(model="meta/llama-3.1-405b-instruct", temperature=0)
llm_model = init_chat_model('llama-3.3-70b-specdec', model_provider='groq', temperature=0)
prompt_template = PromptTemplate(
    input_variables=["context", "user_query", "favorite_places", "visitor_type", "num_days", "budget"],
    template="""You are a helpful travel planner AI.
Use the context below, which contains information about ticket prices, place descriptions, restaurant details, and art gallery information.

Context:
{context}

User Query:
{user_query}

Additional Preferences:
- Favorite types of places: {favorite_places}
- Visitor type: {visitor_type} (e.g., Egyptian, Egyptian student, Foreign, or foreign student)
- Number of travel days: {num_days}
- Overall budget for all days: {budget} EGP
- Exclude hotels from the plan.
- Ensure that the itinerary includes at least 3 meals per day.

Based on the above, return a detailed {num_days}-day travel itinerary with approximate costs and suggestions. If some details are missing, make reasonable assumptions and indicate them.
"""
)

# Structured Output Schema
json_schema = {
    "title": "TravelItinerary",
    "description": "A structured travel itinerary for the user.",
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "description": "List of days with planned activities.",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "Theme of the Day or Day label, e.g., 'Day 1'"},
                    "activities": {
                        "type": "array",
                        "description": "Activities planned for the day.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string", "description": "Time of the activity"},
                                "activity": {"type": "string", "description": "Name of the activity"},
                                "location": {"type": "string", "description": "Location name"},
                                "price_range": {"type": "string", "description": "Price range or cost"},
                            },
                            "required": ["time", "activity", "location"]
                        }
                    },
                    "approximate_cost": {"type": "string", "description": "Total cost for the day"}
                },
                "required": ["day", "activities", "approximate_cost"]
            }
        },
        "total_approximate_cost": {
            "type": "string",
            "description": "Total cost for the trip"
        },
        "notes": {
            "type": "string",
            "description": "Any additional notes or assumptions"
        }
    },
    "required": ["days", "total_approximate_cost"]
}

class Activity(BaseModel):
    time: str = Field(..., description="Time of the activity")
    activity: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location name")
    price_range: Optional[str] = Field(None, description="Price range or cost")

class DayPlan(BaseModel):
    day: str = Field(..., description="Theme of the Day or Day label, e.g., 'Day 1'")
    activities: List[Activity] = Field(..., description="Activities planned for the day")
    approximate_cost: str = Field(..., description="Total cost for the day")

class TravelItinerary(BaseModel):
    days: List[DayPlan] = Field(..., description="List of days with planned activities")
    total_approximate_cost: str = Field(..., description="Total cost for the trip")
    notes: Optional[str] = Field(None, description="Any additional notes or assumptions")

structured_llm = llm_model.with_structured_output(json_schema)

# Generate Travel Plan
def generate_travel_plan(user_query, favorite_places, visitor_type, num_days, budget):
    docs = retriever.invoke(user_query)
    context_text = "\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(
        context=context_text,
        user_query=user_query,
        favorite_places=favorite_places,
        visitor_type=visitor_type,
        num_days=num_days,
        budget=budget
    )
    response = structured_llm.invoke(prompt)
    return response

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Generate a travel plan based on user input.")
    parser.add_argument('--user_query', type=str, required=True, help="User's travel query")
    parser.add_argument('--favorite_places', type=str, required=True, help="User's favorite types of places")
    parser.add_argument('--visitor_type', type=str, required=True, help="Visitor type (e.g., Foreign, Egyptian)")
    parser.add_argument('--num_days', type=str, required=True, help="Number of travel days")
    parser.add_argument('--budget', type=str, required=True, help="Overall budget in EGP")

    args = parser.parse_args()

    # Generate the travel plan
    travel_plan = generate_travel_plan(
        user_query=args.user_query,
        favorite_places=args.favorite_places,
        visitor_type=args.visitor_type,
        num_days=args.num_days,
        budget=args.budget
    )

    # Convert the response to JSON and print to stdout
    print(json.dumps(travel_plan, indent=2))

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()  # End timer
    print(f"Execution time: {end_time - start_time:.5f} seconds")