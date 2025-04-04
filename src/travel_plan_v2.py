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
import pickle
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class DataLoader:
    def __init__(self):
        print("📂 Loading datasets...")
        base_path = Path(__file__).parent.resolve()
        data_path = base_path / ".." / "data"
        
        with tqdm(total=2, desc="Loading data files") as pbar:
            self.landmark_prices = pd.read_csv(data_path / "egypt_v0.1.csv")
            pbar.update(1)
            self.places_api_data = pd.read_csv(data_path / "places_details.csv")
            pbar.update(1)
        
        print(f"✅ Loaded {len(self.landmark_prices)} landmarks and {len(self.places_api_data)} places")
        
class DocumentProcessor():
    def __init__(self, landmark_prices: pd.DataFrame, places_api_data: pd.DataFrame, cache_path: str = "documents.pkl"):
        self.cache_path = Path(cache_path)
        self.landmark_prices = landmark_prices
        self.places_api_data = places_api_data
        self.documents = []

    def df_to_document(self) -> List[Document]:
        total_rows = len(self.landmark_prices) + len(self.places_api_data)
        
        with tqdm(total=total_rows, desc="Processing documents") as pbar:
            # Process landmark prices
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
                pbar.update(1)

            # Process places API data
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
                pbar.update(1)
                
        return self.documents
    
    def load_or_process_documents(self) -> List[Document]:
        if self.cache_path.exists():
            print("📂 Loading cached documents...")
            with tqdm(total=1, desc="Loading document cache") as pbar:
                with open(self.cache_path, "rb") as f:
                    self.documents = pickle.load(f)
                pbar.update(1)
            print(f"✅ Loaded {len(self.documents)} documents from cache")
        else:
            print("🔄 Processing documents from raw data...")
            self.documents = self.df_to_document()
            
            print("💾 Saving documents to cache...")
            with tqdm(total=1, desc="Saving document cache") as pbar:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.documents, f)
                pbar.update(1)
            print(f"✅ Processed and cached {len(self.documents)} documents")
            
        return self.documents
        
class VectorStoreManager():
    def __init__(self, documents: List[Document], provider=HuggingFaceEmbeddings, path='faiss_mpnetv2_v1.0', embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = provider(model_name=embedding_model)
        self.path = Path(path)
        self.documents = documents
        
    def get_retriever(self):
        print(f"📁 Checking for FAISS index at: {self.path}")
        if not self.path.exists():
            print("🚧 Index not found. Creating...")
            
            # Create progress indicator for index creation
            total_docs = len(self.documents)
            with tqdm(total=total_docs, desc="Creating FAISS index") as pbar:
                # We'll use a wrapper to update the progress bar
                def progress_update(batch_size=100):
                    pbar.update(batch_size)
                
                # Create chunks to show progress
                chunk_size = 100
                chunked_docs = [self.documents[i:i + chunk_size] for i in range(0, len(self.documents), chunk_size)]
                
                # Start with first chunk
                vectorstore = FAISS.from_documents(chunked_docs[0], self.embeddings)
                progress_update(len(chunked_docs[0]))
                
                # Add remaining chunks
                for chunk in chunked_docs[1:]:
                    vectorstore.add_documents(chunk)
                    progress_update(len(chunk))
                    
                # Save the index
                with tqdm(total=1, desc="Saving FAISS index") as save_pbar:
                    vectorstore.save_local(self.path)
                    save_pbar.update(1)
                    
            print("✅ FAISS index created and saved.")
        else:
            print("📂 FAISS index exists. Loading...")
            with tqdm(total=1, desc="Loading FAISS index") as pbar:
                vectorstore = FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)
                pbar.update(1)
            print("✅ FAISS index loaded.")

        return vectorstore.as_retriever(search_kwargs={"k": 50})

class Activity(BaseModel):
    time: str = Field(..., description="Time of the activity")
    activity: str = Field(..., description="Name of the activity")
    location: str = Field(..., description="Location name")
    price_range: Optional[str] = Field(None, description="Price range or cost")

class Day(BaseModel):
    day: str = Field(..., description="Theme of the Day or Day label, e.g., 'Day 1'")
    activities: List[Activity] = Field(..., description="Activities planned for the day")
    approximate_cost: str = Field(..., description="Total cost for the day")

class TravelItinerary(BaseModel):
    days: List[Day] = Field(..., description="List of days with planned activities")
    total_approximate_cost: str = Field(..., description="Total cost for the trip")
    notes: Optional[str] = Field(None, description="Any additional notes or assumptions")


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
        print(f"🔄 Initializing LLM: {model_name} via {provider}")
        self.llm = self.initialize_llm()
        print(f"✅ LLM ready: {model_name}")
        
    def initialize_llm(self):
        if self.provider == 'groq' or self.provider == 'google-genai':
            return init_chat_model(self.model_name, model_provider=self.provider, temperature=self.temperature).with_structured_output(self.json_schema)
        elif self.provider == 'nvidia':
            return ChatNVIDIA(self.model_name, temperature=self.temperature).with_structured_output(self.json_schema)
        else:
            raise ValueError('Unsupported model provider')
        
    def travel_plan(self, retriever, city, favorite_places, visitor_type, num_days, budget, callbacks=None):
        """
        Extended version of travel_plan that accepts a callbacks parameter for token tracking.
        """
        user_query = f'Plan a {num_days}-day trip in {city} with visits to {favorite_places}, and dining options.'
        
        print("🔍 Retrieving relevant documents...")
        with tqdm(total=1, desc="Vector search") as pbar:
            docs = retriever.invoke(user_query)
            pbar.update(1)
        print(f"✅ Retrieved {len(docs)} relevant documents")
        
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
        
        print("🧠 Generating travel itinerary...")
        
        # If no callbacks were provided, use our progress callback
        if not callbacks:
            callbacks = [BaseCallbackHandler]
            
        return self.llm.invoke(prompt, config={"callbacks": callbacks})
    
    
if __name__ == '__main__':
    start_time = time.time()  # Start timer
    print("🚀 Travel Planner v2 - With Progress Tracking")

    parser = argparse.ArgumentParser(description="Generate a travel plan based on user input.")
    parser.add_argument('--city', type=str, required=True, help="User's travel City")
    parser.add_argument('--favorite_places', type=str, required=True, help="User's favorite types of places")
    parser.add_argument('--visitor_type', type=str, required=True, help="Visitor type (e.g., Foreign, Egyptian)")
    parser.add_argument('--num_days', type=str, required=True, help="Number of travel days")
    parser.add_argument('--budget', type=str, required=True, help="Overall budget in EGP")

    args = parser.parse_args()
    
    data = DataLoader()
    document_processor = DocumentProcessor(data.landmark_prices, data.places_api_data)
    documents = document_processor.load_or_process_documents()
    
    vector_store = VectorStoreManager(documents=documents)
    retriever = vector_store.get_retriever()
    
    llm_manager = LLMService("qwen-qwq-32b")

    travel_plan = llm_manager.travel_plan(retriever, args.city, args.favorite_places, args.visitor_type, args.num_days, args.budget)
    
    print("\n📋 Generated Travel Plan:")
    print(json.dumps(travel_plan, indent=4))
    
    end_time = time.time()  # End timer
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")