import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from typing import List, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
import argparse
import time

class DataLoader:
    def __init__(self):
        self.landmark_prices = pd.read_csv(os.path.abspath('../Travel-Agent/data/egypt_v0.1.csv'))
        self.places_api_data = pd.read_csv(os.path.abspath('../Travel-Agent/data/places_details.csv'))

class DocumentProcessor():
    def __init__(self, landmark_prices: pd.DataFrame, places_api_data: pd.DataFrame):
        self.landmark_prices = landmark_prices
        self.places_api_data = places_api_data
        self.documents = []


    def df_to_document(self) -> List[Document]:
        for _, row in self.landmark_prices.iterrows():
            text = f"""
            Governorate: {row.get('Governorate/City', 'N/A')}
            Site: {row.get('Place', 'N/A')}
            Egyptian Ticket: {row.get('Egyptian', 'N/A')} EGP
            Egyptian Student Ticket: {row.get('EgyptianStudent', 'N/A')} EGP
            Foreign Ticket: {row.get('Foreign', 'N/A')} EGP
            Foreign Student Ticket: {row.get('ForeignStudent', 'N/A')} EGP
            Visiting Times: {row.get('VisitingTimes', 'N/A')}
            """
            self.documents.append(Document(page_content=text, metadata={"source": 'landmark_prices'}))

        for _, row in self.places_api_data.iterrows():
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
            self.documents.append(Document(page_content=text, metadata={"source": 'Places_api', 'Type': f"{row['primaryTypeDisplayName.text']}", 'city': f"{row['formattedAddress']}"}))
            
        return self.documents
        
class VectorStoreManager():
    def __init__(self, documents: List[Document], provider=HuggingFaceEmbeddings, path='faiss_mpnetv2_v1.0', embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = provider(model_name=embedding_model)
        self.path = Path(path)
        self.documents = documents
        
    def get_retriever(self):
        if not self.path.exists():
            vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            vectorstore.save_local(self.path)
            
        else:
            vectorstore = FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)


        return vectorstore.as_retriever(search_kwargs={"k": 50})
    

class LLMService:
    def __init__(self, model_name, provider='groq', temperature=0):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.json_schema = {
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
                                "des3                                           cription": "Activities planned for the day.",
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
        self.llm =self.initialize_llm()
        
    def initialize_llm(self):
        if self.provider == 'groq' or 'google_genai':
            return init_chat_model(self.model_name, model_provider= self.provider, temperature = self.temperature).with_structured_output(self.json_schema)
        elif self.provider == 'nvidia':
            return ChatNVIDIA(self.model, temperature = self.temperature).with_structured_output(self.json_schema)
        else:
            raise ValueError('Unsupported model provider')

        
               
    def travel_plan(self, retriever, city, favorite_places, visitor_type, num_days, budget):
        
        user_query = f'Plan a {num_days}-day trip in {city} with visits to {favorite_places}, and dining options.'
        docs = retriever.invoke(user_query)
        context_text = "\n".join([doc.page_content for doc in docs])
        
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
        
        prompt = prompt_template.format(
            context=context_text,
            user_query=user_query,
            favorite_places=favorite_places,
            visitor_type=visitor_type,
            num_days=num_days,
            budget=budget
        )
        return self.llm.invoke(prompt)
    
    
if __name__ == '__main__':
    # load_dotenv()
    start_time = time.time()  # Start timer

    parser = argparse.ArgumentParser(description="Generate a travel plan based on user input.")
    parser.add_argument('--city', type=str, required=True, help="User's travel query")
    parser.add_argument('--favorite_places', type=str, required=True, help="User's favorite types of places")
    parser.add_argument('--visitor_type', type=str, required=True, help="Visitor type (e.g., Foreign, Egyptian)")
    parser.add_argument('--num_days', type=str, required=True, help="Number of travel days")
    parser.add_argument('--budget', type=str, required=True, help="Overall budget in EGP")

    args = parser.parse_args()
    
    
    
    data = DataLoader()
      
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.df_to_document()
    
    v = VectorStoreManager(documents)
    retriever = v.get_retriever()
    
    llm_manager = LLMService('llama-3.3-70b-specdec')
    
    # user_query = "Plan a 3-day trip in Luxor with visits to cultural sites, art galleries, and dining options."
    # favorite_places = "Cultural sites, historical landmarks, art galleries"
    # visitor_type = "Foreign"
    # num_days = "3"
    # budget = "5000"
    
    travel_plan = llm_manager.travel_plan(retriever, args.city, args.favorite_places, args.visitor_type, args.num_days, args.budget)
    print(json.dumps(travel_plan, indent=2))
    
    end_time = time.time()  # End timer
    print(f"Execution time: {end_time - start_time:.5f} seconds")