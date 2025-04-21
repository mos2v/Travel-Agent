import pandas as pd
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
import argparse
import time
import pickle
from langchain.callbacks.base import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import torch

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Rag_travel_planner_v3.0.0"



# Define color scheme for different operation types
PROGRESS_COLORS = {
    'data_loading': '#00FF00',    # Blue for data loading
    'processing': '#13D4D4',      # Green for data processing
    'vectorstore': '#C71585',     # Purple for vector operations
    'default': '#00cc66',         # Default green
}



def pretty_progress_bar(iterable=None, total=None, desc=None, operation_type='default', **kwargs):
    """
    Creates a prettier progress bar with enhanced styling and color based on operation type.
    
    Parameters:
    - iterable: Optional iterable to decorate with a progressbar
    - total: Override the iterable length if iterable size is unknown
    - desc: Progress bar description prefix
    - operation_type: Type of operation for color coding
                     ('data_loading', 'processing', 'vectorstore', 'llm', or 'default')
    - **kwargs: Additional kwargs for tqdm
    """
    # Get color based on operation type
    color = PROGRESS_COLORS.get(operation_type, PROGRESS_COLORS['default'])
    
    # Set default styling for prettier bars
    custom_kwargs = {
        'colour': color,           # Color based on operation type
        'bar_format': '{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        'ascii': False,             # Use Unicode blocks instead of ASCII for the bar
        'ncols': 100,               # Fixed width for consistent appearance
        'leave': True,              # Keep the progress bar after completion
        'dynamic_ncols': True,      # Adapt to terminal size changes
        'smoothing': 0.3,           # Smoother animation
    }
    
    # Add operation type icons based on operation type
    icons = {
        'data_loading': '📂',
        'processing': '⚙️',
        'vectorstore': '🔍',
        'llm': '🧠',
        'default': '✨',
    }
    icon = icons.get(operation_type, icons['default'])
    
    # If description is provided, add emoji and formatting
    if desc:
        desc = f"{icon} {desc} "
    
    # Override defaults with any user-provided kwargs
    custom_kwargs.update(kwargs)
    
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        **custom_kwargs
    )


class DataLoader:
    def __init__(self):
        print("📂 Loading datasets...")
        base_path = Path(__file__).parent.resolve()
        data_path = base_path / ".." / "data"
        
        with pretty_progress_bar(total=2, desc="Loading data files", operation_type='data_loading') as pbar:
            self.landmark_prices = pd.read_csv(data_path / "egypt_v0.1.csv")
            pbar.update(1)
            self.places_api_data = pd.read_csv(data_path / "places_details_v1.csv")
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
        
        with pretty_progress_bar(total=total_rows, desc="Processing documents", operation_type='processing') as pbar:
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
            with pretty_progress_bar(total=1, desc="Loading document cache", operation_type='data_loading') as pbar:
                with open(self.cache_path, "rb") as f:
                    self.documents = pickle.load(f)
                pbar.update(1)
            print(f"✅ Loaded {len(self.documents)} documents from cache")
        else:
            print("🔄 Processing documents from raw data...")
            self.documents = self.df_to_document()
            
            print("💾 Saving documents to cache...")
            with pretty_progress_bar(total=1, desc="Saving document cache", operation_type='data_loading') as pbar:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.documents, f)
                pbar.update(1)
            print(f"✅ Processed and cached {len(self.documents)} documents")
            
        return self.documents    
        
        
class VectorStoreManager():
    def __init__(self, documents: List[Document] = None, provider=HuggingFaceEmbeddings, path='faiss_e5large_v1.0', embedding_model="intfloat/multilingual-e5-large-instruct"):
        self.embeddings = provider(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={'normalize_embeddings': True}
            )
        self.path = Path(path)
        self.documents = documents
        
    def get_retriever(self):
        print(f"📁 Checking for FAISS index at: {self.path}")
        if not self.path.exists():
            print("🚧 Index not found. Creating...")
            
            # Create progress indicator for index creation
            total_docs = len(self.documents)
            with pretty_progress_bar(total=total_docs, desc="Creating FAISS index", operation_type='vectorstore') as pbar:
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
                with pretty_progress_bar(total=1, desc="Saving FAISS index", operation_type='vectorstore') as save_pbar:
                    vectorstore.save_local(self.path)
                    save_pbar.update(1)
                    
            print("✅ FAISS index created and saved.")
        else:
            print("📂 FAISS index exists. Loading...")
            with pretty_progress_bar(total=1, desc="Loading FAISS index", operation_type='vectorstore') as pbar:
                vectorstore = FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)
                pbar.update(1)
            print("✅ FAISS index loaded.")

        return vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={
                "k": 50,                     
                "score_threshold": 0.5})

class LLMService:
    def __init__(self, model_name, provider='groq', temperature: float = 0.0):
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
        print(f"🔄 Initializing LLM: {model_name} via {provider} with temperature {temperature}")
        self.llm = self.initialize_llm()
        print(f"✅ LLM ready: {model_name}")
        
    def initialize_llm(self):
        if self.provider == 'groq':
            return ChatGroq(model=self.model_name, temperature=self.temperature).with_structured_output(self.json_schema)
        elif self.provider == 'nvidia':
            return ChatNVIDIA(model=self.model_name, temperature=self.temperature).with_structured_output(self.json_schema)
        elif self.provider == "mistralai":
            return ChatMistralAI(model_name=self.model_name, temperature=self.temperature).with_structured_output(self.json_schema)
        elif self.provider == 'google-genai':
            return ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature).with_structured_output(self.json_schema)
        else: 
                raise ValueError('Unsupported model provider')
            
            
    def _generate_enhanced_query(self, destination, interests, visitor_type):
        """
        Generate multiple focused queries to ensure balanced retrieval of different place types.
        
        Args:
            destination (str): The main destination city for the trip
            interests (list): List of specific interests like ["temples", "museums", "local cuisine"]
            visitor_type (str): Type of visitor (Egyptian, Foreign, etc.)
        Returns:
            dict: Dictionary with different specialized queries
        """
        # Format the interests as a comma-separated string
        if isinstance(interests, list):
            interests_str = ", ".join(interests)
        else:
            interests_str = interests
        
        # Create specialized queries for different categories
        queries = {
            "attractions": f"""
                Find tourist attractions, historical landmarks, and museums ONLY in {destination}, Egypt.
                Must be located within {destination} city limits. Exclude attractions from other cities.
            """,
            
            "restaurants": f"""
                Find restaurants, cafes, and dining options ONLY in {destination}, Egypt.
                Must be located within {destination} city limits. Exclude places from other cities.
            """,
            
            "historical_sites": f"""
                Find historical sites and monuments ONLY in {destination}, Egypt.
                Must be located within {destination} city limits. Exclude sites from other cities.
            """,
            
            "prices": f"""
                Find ticket prices and visiting hours for attractions in {destination}, Egypt.
                Must be specifically for {destination} attractions.
            """
        }
        
        # Add interests query only if interests were provided
        if interests_str and interests_str != "popular attractions":
            queries["interests"] = f"""
                Find {interests_str} ONLY in {destination}, Egypt.
                Must be located within {destination} city limits.
            """
        
        # Clean up whitespace in all queries
        for key in queries:
            queries[key] = queries[key].strip()
            
        return queries

    def travel_plan(self, retriever, city, favorite_places, visitor_type, num_days, budget, callbacks=None):
        """
        Extended version of travel_plan that retrieves balanced content across categories 
        while eliminating duplicates and ensuring city-specific results.
        """
        user_query = f'Plan a {num_days}-day trip in {city} with visits to {favorite_places}, and dining options.'
        
        # Standardize city name for consistent matching
        city_lower = city.lower().strip()
        
        # Generate specialized queries for different types of places
        query_dict = self._generate_enhanced_query(city, favorite_places, visitor_type)
        
        print("🔍 Retrieving relevant documents...")
        all_docs = []
        category_docs = {}
        
        # Track document content hashes to avoid duplicates during retrieval
        doc_content_hashes = set()
        
        # City variations to check for proper filtering
        city_variations = [city_lower, f"{city_lower},", f"{city_lower} governorate"]
        
        # Helper function to check if a document mentions the target city
        def is_city_match(doc):
            if not hasattr(doc, 'page_content'):
                return False
                
            page_content_lower = doc.page_content.lower()
            
            # 1. Check if doc is from landmark_prices with matching city
            if hasattr(doc, 'metadata') and doc.metadata.get('source') == 'landmark_prices':
                # Extract Governorate/City from content
                if "governorate: " in page_content_lower:
                    doc_city = None
                    lines = page_content_lower.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("governorate: ") or line.startswith("governorate/city: "):
                            doc_city = line.split(":", 1)[1].strip().lower()
                            break
                            
                    if doc_city and any(var == doc_city for var in city_variations):
                        return True
                return False
                
            # 2. Check metadata for explicit city information
            if hasattr(doc, 'metadata'):
                # Check 'city' field
                if 'city' in doc.metadata and isinstance(doc.metadata['city'], str):
                    metadata_city = doc.metadata['city'].lower()
                    if any(var in metadata_city for var in city_variations):
                        return True
                        
                # Check 'formattedAddress' field
                if 'formattedAddress' in doc.metadata and isinstance(doc.metadata['formattedAddress'], str):
                    address = doc.metadata['formattedAddress'].lower()
                    if any(var in address for var in city_variations):
                        return True
                        
            # 3. Check content for city mention in specific context
            # More strict content checking to avoid false positives
            if "place location:" in page_content_lower:
                location_line = None
                lines = page_content_lower.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("place location:"):
                        location_line = line
                        break
                        
                if location_line and any(var in location_line for var in city_variations):
                    return True
                    
            # Document doesn't match city criteria
            return False
        
        # Helper function to get content hash for deduplication
        def get_content_key(doc):
            """Generate a unique string key for a document to use for deduplication"""
            if not hasattr(doc, 'page_content'):
                return None
                
            # Clean the content for more effective deduplication
            content = doc.page_content.strip()
            
            # For landmark price documents, extract key identifying information
            if hasattr(doc, 'metadata') and doc.metadata.get('source') == 'landmark_prices':
                site_name = None
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("Site:"):
                        site_name = line.split(":", 1)[1].strip()
                        break
                if site_name:
                    return f"landmark-{site_name}"
                    
            # For place documents, extract name as identifying information
            if "place name:" in content.lower():
                place_name = None
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith("place name:"):
                        place_name = line.split(":", 1)[1].strip()
                        break
                if place_name:
                    return f"place-{place_name}"
                    
            # Default to string hash of entire content
            return f"content-{hash(content)}"
        
        with pretty_progress_bar(total=len(query_dict), desc="Vector search", operation_type='vectorstore') as pbar:
            for category, query in query_dict.items():
                # Get documents for each category
                docs = retriever.invoke(query)
                filtered_docs = []
                
                for doc in docs:
                    content_key = get_content_key(doc)
                    
                    # Skip if we've seen this document before or it's invalid
                    if not content_key or content_key in doc_content_hashes:
                        continue
                    
                    # Only include documents that match the city
                    if is_city_match(doc):
                        doc_content_hashes.add(content_key)
                        filtered_docs.append(doc)
                
                # Add category-specific filtered docs to results
                if filtered_docs:
                    category_docs[category] = filtered_docs
                    all_docs.extend(filtered_docs)
                
                pbar.update(1)
        
        # Special handling for restaurant category - ensure we have enough restaurants
        if 'restaurants' not in category_docs or len(category_docs.get('restaurants', [])) < 5:
            # Create more specific restaurant queries for the city
            restaurant_queries = [
                f"restaurants in {city} Egypt",
                f"cafes in {city} Egypt",
                f"dining options in {city} Egypt"
            ]
            
            print("🍽️ Finding additional dining options...")
            with pretty_progress_bar(total=len(restaurant_queries), desc="Restaurant search", operation_type='vectorstore') as pbar:
                for query in restaurant_queries:
                    docs = retriever.invoke(query)
                    
                    for doc in docs:
                        content_key = get_content_key(doc)
                        
                        # Skip if we've seen this document before or it's invalid
                        if not content_key or content_key in doc_content_hashes:
                            continue
                        
                        # Only include restaurants that match the city
                        if is_city_match(doc):
                            doc_content_hashes.add(content_key)
                            
                            # Add to restaurants category and all_docs
                            if 'restaurants' not in category_docs:
                                category_docs['restaurants'] = []
                            category_docs['restaurants'].append(doc)
                            all_docs.append(doc)
                    
                    pbar.update(1)
        
        # Organize documents with category labels for better context
        organized_docs = []
        
        # Start with pricing information (landmark prices) - limited to 10 most relevant
        landmark_docs = [doc for doc in all_docs if 
                        hasattr(doc, 'metadata') and 
                        doc.metadata.get('source') == 'landmark_prices']
        
        if landmark_docs:
            organized_docs.append("--- TICKET PRICES AND VISITING HOURS ---")
            for doc in landmark_docs[:10]:  
                organized_docs.append(doc.page_content)
        
        # Add restaurant information - limited to 10 most relevant
        restaurant_docs = category_docs.get('restaurants', [])
        if restaurant_docs:
            organized_docs.append("\n--- RESTAURANTS AND CAFES ---")
            for doc in restaurant_docs[:12]: 
                organized_docs.append(doc.page_content)
        
        # Add tourist attractions - limited to 10 most relevant using manual deduplication
        # NOT using set() on Document objects since they're not hashable
        attraction_docs = []
        attraction_keys = set()
        
        # Combine attractions and historical sites
        combined_attraction_docs = category_docs.get('attractions', []) + category_docs.get('historical_sites', [])
        
        # Manual deduplication
        for doc in combined_attraction_docs:
            content_key = get_content_key(doc)
            if content_key and content_key not in attraction_keys:
                attraction_keys.add(content_key)
                attraction_docs.append(doc)
        
        if attraction_docs:
            organized_docs.append("\n--- ATTRACTIONS AND HISTORICAL SITES ---")
            for doc in attraction_docs[:10]:  # Limit to top 10
                organized_docs.append(doc.page_content)
        
        # Add interest-specific places - limited to 5 most relevant
        interest_docs = category_docs.get('interests', [])
        if interest_docs:
            organized_docs.append("\n--- PLACES MATCHING YOUR INTERESTS ---")
            for doc in interest_docs[:8]:  # Limit to top 5
                organized_docs.append(doc.page_content)
        
        # Combine organized documents into context
        context_text = "\n\n".join(organized_docs)
        
        # Log statistics about retrieved documents
        print(f"✅ Retrieved {len(doc_content_hashes)} unique documents for {city}:")
        print(f"   - {len(landmark_docs)} pricing/visiting hours documents")
        print(f"   - {len(restaurant_docs)} restaurant and cafe documents")
        print(f"   - {len(attraction_docs)} attraction documents")
        print(f"   - {len(interest_docs)} interest-specific documents")
        
        
        budget_conscious_prompt = PromptTemplate(
            input_variables=["context", "user_query", "favorite_places", "visitor_type", "num_days", "budget", "city"],
            template="""You are an expert Egyptian travel planner with extensive knowledge of historical sites, cultural attractions, local cuisine, and hidden gems across Egypt. Your task is to create a detailed, balanced {num_days}-day itinerary for {visitor_type} visitors to {city}, Egypt.
            
            ### AVAILABLE INFORMATION:
            {context}

            ### USER REQUEST:
            {user_query}

            ### USER PREFERENCES:
            - City/Destination: {city} (ONLY include places in this city)
            - Favorite types of places: {favorite_places}
            - Visitor category: {visitor_type} (Affects ticket pricing)
            - Trip duration: {num_days} days
            - MAXIMUM TOTAL BUDGET: {budget} EGP for the entire trip (THIS IS A HARD CONSTRAINT)

            ### DETAILED INSTRUCTIONS:
            1. CITY RESTRICTION (HIGHEST PRIORITY):
            - ONLY include attractions, restaurants, and activities in {city}
            - DO NOT include any places from other cities or regions
            - Verify each recommendation against the provided context data

            2. BALANCE OF ACTIVITIES (VERY IMPORTANT):
            - Include a balanced mix of attractions AND dining experiences each day
            - Each day must include:
            * 1-2 major attractions or historical sites
            * 3 meals at different restaurants/cafes (breakfast, lunch, dinner)
            * At least one activity related to user's specified favorite places

            3. BUDGET MANAGEMENT (HIGHEST PRIORITY):
            - The total cost MUST NOT EXCEED {budget} EGP under any circumstances
            - Use EXACT TICKET PRICES from the "TICKET PRICES AND VISITING HOURS" section when available
            - Use realistic price estimates for restaurants based on the "RESTAURANTS AND CAFES" section
            - Reserve 10% of the budget for transportation between sites
            - If the budget is tight, prioritize must-see attractions over secondary experiences

            4. ATTRACTIONS SELECTION:
            - Select attractions directly mentioned in the retrieved context
            - Use the exact ticket prices provided for {visitor_type} visitors
            - Consider free attractions (those with 0 EGP ticket price) to maximize budget value
            - Plan visits according to the provided opening hours/visiting times
            - Group nearby attractions to minimize transportation needs

            5. DINING RECOMMENDATIONS:
            - Select restaurants and cafes specifically mentioned in the "RESTAURANTS AND CAFES" section
            - Include estimated meal costs based on price information provided (or conservative estimates)
            - Vary dining experiences between traditional Egyptian cuisine and other options
            - Consider price levels (moderate, inexpensive) noted in the restaurant data

            6. DAILY STRUCTURE:
            - Organize each day in chronological order (morning to evening)
            - Allow sufficient time at major attractions (2-3 hours minimum)
            - Include specific visit times that align with attraction opening hours
            - Account for travel time between locations

            7. BUDGET TRACKING:
            - Itemize each expense (attraction tickets, meals, etc.)
            - Provide a daily subtotal at the end of each day
            - Maintain a running cumulative total throughout the itinerary
            - Ensure the final total stays under {budget} EGP

            DO NOT include hotels or accommodations in your plan.
            DO NOT exceed the total budget provided - this is a strict requirement.
            DO NOT recommend places not listed in the provided context.
            DO NOT include places from cities other than {city}.
            DO include a diverse mix of attractions and dining options.

            Your response must follow the structured format required by the JSON schema, with complete details for each day's activities and accurate cost tracking.
            """
        )
        
        prompt = budget_conscious_prompt.format(
            context=context_text,
            user_query=user_query,
            favorite_places=favorite_places,
            visitor_type=visitor_type,
            num_days=num_days,
            budget=budget,
            city=city
        )
        
        print("🧠 Generating travel itinerary...")
        
        # If no callbacks were provided, use our progress callback
        if not callbacks:
            callbacks = [BaseCallbackHandler]
            
        return self.llm.invoke(prompt, config={"callbacks": callbacks})

if __name__ == '__main__':
    start_time = time.time() 
    print("🚀 Travel Planner v3")
    
    load_dotenv()
    
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
    
    # llm_manager = LLMService("nvidia/llama-3.1-nemotron-ultra-253b-v1", provider="nvidia")
    # llm_manager = LLMService("nvidia/llama-3.3-nemotron-super-49b-v1", provider="nvidia")
    # llm_manager = LLMService("mistral-large-latest", provider="mistralai")
    # llm_manager = LLMService("meta/llama-4-maverick-17b-128e-instruct", provider='nvidia')
    llm_manager = LLMService("gemini-2.0-flash", provider="google-genai", temperature=0.4)
    # llm_manager = LLMService("gemini-2.5-flash-preview-04-17", provider="google-genai", temperature=0.4)
    # llm_manager = LLMService("meta-llama/llama-4-maverick-17b-128e-instruct", provider='groq')
    
    travel_plan = llm_manager.travel_plan(retriever, args.city, args.favorite_places, args.visitor_type, args.num_days, args.budget)
    
    print("\n📋 Generated Travel Plan:")
    print(json.dumps(travel_plan, indent=4))
    
    end_time = time.time()  # End timer
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")    