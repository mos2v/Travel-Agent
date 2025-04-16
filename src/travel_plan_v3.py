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
# Define color scheme for different operation types
PROGRESS_COLORS = {
    'data_loading': '#13D4D4',    # Blue for data loading
    'processing': '#00FF00',      # Green for data processing
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
        self.embeddings = provider(model_name=embedding_model)
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
    def __init__(self, model_name, provider='groq', temperature: float = 0):
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
        Generate an enhanced query that will retrieve more relevant documents from the vector store,
        with special focus on dining options and food experiences.
        
        Args:
            destination (str): The main destination city for the trip
            interests (list): List of specific interests like ["temples", "museums", "local cuisine"]
            visitor_type (str): Type of visitor (Egyptian, Foreign, etc.)
        Returns:
            str: Enhanced query for better retrieval
        """
        # Format the interests as a comma-separated string
        interests_str = ", ".join(interests)
        
        # Add food-related keywords to interests if not already present
        food_terms = ["restaurants", "cafes", "local cuisine", "dining", "food"]
        food_interests = [term for term in food_terms if term not in interests_str.lower()]
        if food_interests:
            interests_str += f", {', '.join(food_interests)}"
        
        # Construct a more specific query with keywords that match document content
        query = f"""
        Detailed travel plan for {destination} Egypt focusing on {interests_str}.
        I need comprehensive information about:
        
        ATTRACTIONS:
        1. Ticket prices for {visitor_type} visitors
        2. Opening hours and visiting times for attractions
        3. Cultural sites and museums in {destination}
        
        DINING OPTIONS (IMPORTANT):
        1. Popular restaurants in {destination} with price range between 100-300 EGP
        2. Coffee shops and cafes in {destination}
        3. Traditional Egyptian dining establishments
        4. Food markets and street food locations
        5. Breakfast locations open in the morning
        6. Lunch restaurants with good ratings
        7. Dinner options that serve authentic Egyptian cuisine
        8. Dessert places and sweet shops
        9. Specialty food items in {destination}
        10. Restaurant rating information and visitor reviews
        11. Restaurants with Nile views or special settings
        12. Seafood restaurants in {destination}
        
        Please include specific names, locations, price ranges, and opening hours for all food establishments.
        """
        return query.strip()
    
    def travel_plan(self, retriever, city, favorite_places, visitor_type, num_days, budget, callbacks=None):
        """
        Extended version of travel_plan that accepts a callbacks parameter for token tracking.
        """
        user_query = f'Plan a {num_days}-day trip in {city} with visits to {favorite_places}, and dining options.'
        
        retriever_query = self._generate_enhanced_query(city, favorite_places, visitor_type)
        print("🔍 Retrieving relevant documents...")
        with pretty_progress_bar(total=1, desc="Vector search", operation_type='vectorstore') as pbar:
            docs = retriever.invoke(retriever_query)
            pbar.update(1)
        print(f"✅ Retrieved {len(docs)} relevant documents")
        
        context_text = "\n".join([doc.page_content for doc in docs])
        
        budget_conscious_prompt = PromptTemplate(
            input_variables=["context", "user_query", "favorite_places", "visitor_type", "num_days", "budget"],
            template="""You are an expert Egyptian travel planner with extensive knowledge of historical sites, cultural attractions, local cuisine, and hidden gems across Egypt. Your task is to create a personalized travel itinerary that STRICTLY ADHERES TO THE BUDGET CONSTRAINTS.

        ### AVAILABLE INFORMATION:
        {context}

        ### USER REQUEST:
        {user_query}

        ### USER PREFERENCES:
        - Favorite types of places: {favorite_places}
        - Visitor category: {visitor_type} (Affects ticket pricing)
        - Trip duration: {num_days} days
        - MAXIMUM TOTAL BUDGET: {budget} EGP for the entire trip (THIS IS A HARD CONSTRAINT)

        ### DETAILED INSTRUCTIONS:
        1. BUDGET MANAGEMENT (HIGHEST PRIORITY):
        - The total cost MUST NOT EXCEED {budget} EGP under any circumstances
        - If necessary, REDUCE THE NUMBER OF ACTIVITIES per day to stay within budget
        - Allocate budget in this order of priority: (1) Must-see attractions, (2) Meals, (3) Secondary attractions
        - Track cumulative costs meticulously throughout the itinerary
        - Reserve 10% of budget for contingencies and transportation between sites

        2. ATTRACTIONS SELECTION:
        - Prioritize attractions that match user's favorite place types AND provide the best value for money
        - For each attraction, include EXACT TICKET PRICES for {visitor_type} visitors
        - If an attraction is expensive but unmissable, compensate by selecting more affordable options for other activities
        - Consider free or low-cost alternatives when possible (e.g., viewpoints, markets, walking tours)

        3. DINING RECOMMENDATIONS:
        - Include 3 meals per day with realistic costs
        - Balance between authentic experiences and budget constraints
        - For expensive destinations, suggest at least one affordable meal option per day
        - Include specific price estimates for each meal

        4. TIME AND ACTIVITY MANAGEMENT:
        - If budget forces reduction in activities, focus on QUALITY over QUANTITY
        - Allow sufficient time at major attractions (2-3 hours minimum)
        - Group activities by geographic proximity to reduce transportation costs
        - Include at least one low-cost or free activity each day

        5. BUDGET BREAKDOWN:
        - At the end of each day's itinerary, provide a running total of expenses
        - Clearly itemize all costs in the itinerary
        - If assumptions are made about costs, they should be CONSERVATIVE estimates

        DO NOT include hotels or accommodations in your plan.
        DO NOT exceed the total budget provided - this is a strict requirement.
        DO NOT recommend places without confirmed existence in the provided context.
        DO REDUCE the number of activities rather than exceeding the budget.

        Your response must follow the structured format required by the JSON schema, with complete details for each day's activities and accurate cost tracking.
        """)
        
        prompt = budget_conscious_prompt.format(
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
    
    llm_manager = LLMService("nvidia/llama-3.1-nemotron-ultra-253b-v1", provider="nvidia")
    # llm_manager = LLMService("nvidia/llama-3.3-nemotron-super-49b-v1", provider="nvidia")
    # llm_manager = LLMService("mistral-large-latest", provider="mistralai")
    # llm_manager = LLMService("gemini-2.0-flash", provider="google-genai")

    travel_plan = llm_manager.travel_plan(retriever, args.city, args.favorite_places, args.visitor_type, args.num_days, args.budget)
    
    print("\n📋 Generated Travel Plan:")
    print(json.dumps(travel_plan, indent=4))
    
    end_time = time.time()  # End timer
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")    