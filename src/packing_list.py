import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


# Handle imports for both API usage and direct script execution
try:
    from .travel_plan_v3 import VectorStoreManager, LLMService
except ImportError:
    try:
        from travel_plan_v3 import VectorStoreManager, LLMService
    except ImportError:
        # For direct execution, try importing from src
        import sys
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        from src.travel_plan_v3 import VectorStoreManager, LLMService
class PackingListProcessor:
    def __init__(self, path: str):
        self.df = pd.read_csv(Path(__file__).parent.resolve() / "../data" / path)
        
    def df_to_documents(self) -> list[Document]:
        """
        Convert a DataFrame to a list of Document objects.
        """
        documents = []
        for _, row in self.df.iterrows():
            formatted_row = "\n".join(f"{col}: {row[col]}" for col in self.df.columns)

            documents.append(Document(page_content=formatted_row))
        return documents
    

class PackingListGenerator:
    def __init__(self, model_name, provider='google-genai', temperature: float = 0.3):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.json_output = {
        "title": "PackingList",
        "type": "object",
        "properties": {
            "trip": {
            "description": "Metadata about the itinerary for which this list was generated.",
            "type": "object",
            "properties": {
                "destination":   { "type": "string" },
                "start_date":    { "type": "string" },
                "end_date":      { "type": "string" },
                "nights":        { "type": "integer" }
            },
            "required": ["destination", "nights"]
            },

            "items": {
            "description": "Flat array of individual items to pack.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "name":        { "type": "string" },
                "category":    { "type": "string", "enum": [
                                    "Clothing", "Footwear", "Toiletries",
                                    "Electronics", "Gear", "Accessories",
                                    "Documents", "Medical", "Food"
                                ]},
                "quantity":    { "type": "number" },
                "unit":        { "type": "string" },
                "essential":   { "type": "boolean" },
                "pack_where":  { "type": "string", "enum": [
                                    "wear", "carry_on", "checked_luggage"
                                ]},
                "tags":        { "type": "array", "items": { "type": "string" } },
                "explanation": { "type": "string" },
                "source_doc":  { "type": "string" }
                },
                "required": ["name", "category"]
            }
            }
        },
        "required": ["trip", "items"]
        }

    def initialize_llm(self):
        if self.provider == 'groq':
            return ChatGroq(model=self.model_name, temperature=self.temperature)
        elif self.provider == 'nvidia':
            return ChatNVIDIA(model=self.model_name, temperature=self.temperature)
        elif self.provider == "mistralai":
            return ChatMistralAI(model_name=self.model_name, temperature=self.temperature)      
        elif self.provider == 'google-genai':
            return ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)
        else:
            raise ValueError('Unsupported model provider')

    def generate_query_based_on_user_input(self, user_input: str, travel_plan, city: str = None, start_date: str = None, end_date: str = None) -> str:
        """
        Generate enhanced query based on user input, travel plan, and trip details.
        Optimized for Egypt-specific packing data with tags and categories.
        """
        llm = self.initialize_llm()
        
        # Extract detailed context from travel plan
        activities = []
        locations = []
        activity_types = set()
        
        if isinstance(travel_plan, dict) and 'days' in travel_plan:
            for day in travel_plan['days']:
                if 'activities' in day:
                    for activity in day['activities']:
                        if 'activity' in activity:
                            activity_text = activity['activity']
                            activities.append(activity_text)
                            
                            # Categorize activity types for better search
                            activity_lower = activity_text.lower()
                            if any(term in activity_lower for term in ['temple', 'mosque', 'church', 'religious']):
                                activity_types.add('religious sites')
                            if any(term in activity_lower for term in ['museum', 'gallery', 'exhibition']):
                                activity_types.add('museums')
                            if any(term in activity_lower for term in ['beach', 'sea', 'swimming', 'snorkel', 'dive']):
                                activity_types.add('beach activities')
                            if any(term in activity_lower for term in ['desert', 'safari', 'camel', 'dune']):
                                activity_types.add('desert tours')
                            if any(term in activity_lower for term in ['market', 'bazaar', 'souk', 'shopping']):
                                activity_types.add('markets shopping')
                            if any(term in activity_lower for term in ['walk', 'tour', 'sightseeing', 'explore']):
                                activity_types.add('walking sightseeing')
                            if any(term in activity_lower for term in ['cruise', 'boat', 'nile']):
                                activity_types.add('water activities')
                                
                        if 'location' in activity:
                            location_text = activity['location']
                            locations.append(location_text)
        
        # Determine season and month with more detail
        season_info = ""
        temperature_context = ""
        if start_date and end_date:
            from datetime import datetime
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                month = start_dt.month
                month_name = start_dt.strftime('%B')
                
                if month in [12, 1, 2]:
                    season = "winter"
                    temperature_context = "mild temperatures, cooler evenings"
                elif month in [3, 4, 5]:
                    season = "spring"
                    temperature_context = "warm pleasant weather"
                elif month in [6, 7, 8]:
                    season = "summer"
                    temperature_context = "very hot, intense sun, high temperatures"
                else:
                    season = "autumn"
                    temperature_context = "warm comfortable weather"
                
                season_info = f"Season: {season}, Month: {month_name}, Weather: {temperature_context}"
            except ValueError:
                season_info = f"Travel dates: {start_date} to {end_date}"
        
        # Build comprehensive context with enhanced details
        context_parts = [
            f"Trip: {user_input}",
            f"Destination: {city}",
            season_info,
            f"Activity types: {', '.join(activity_types)}" if activity_types else "Activities: General sightseeing and cultural exploration",
            f"Specific activities: {', '.join(activities[:8])}" if activities else "",
            f"Key locations: {', '.join(set(locations[:6]))}" if locations else "",
            f"Trip duration: {len(travel_plan.get('days', []))} days" if isinstance(travel_plan, dict) else ""
        ]
        
        context = "\n".join(filter(None, context_parts))
        
        prompt = PromptTemplate(
            input_variables=["context"],
            template="""Generate 4-6 targeted search queries for Egypt packing items based on:

{context}

Database tags include: seasons (summer/winter), cities (Luxor/Cairo/Alexandria), activities (religious/beach/desert/museums), cultural needs (modest clothing), essentials (documents/adapters/health).

Create focused 3-8 word queries covering:
1. Climate/season items
2. Location-specific needs  
3. Activity requirements
4. Cultural/practical essentials

Examples: "summer Egypt sun protection"; "Luxor temple modest clothing"; "Egypt travel documents"

Output 4-6 queries separated by semicolons:"""
        )
        prompt_text = prompt.format(context=context)
        query = llm.invoke(prompt_text, config={"callbacks":[BaseCallbackHandler]})
        return query.content

    def generate_packing_list(self, retriever, travel_plan, city: str = None, start_date: str = None, end_date: str = None):
        """
        Generate a comprehensive packing list based on travel plan and trip details.
        """
        llm = self.initialize_llm().with_structured_output(self.json_output)
        user_input = f"I'm traveling to {city} for {len(travel_plan['days'])} days. What should I pack?"
        
        # Generate enhanced query with all available information
        query_string = self.generate_query_based_on_user_input(user_input, travel_plan, city, start_date, end_date)
        print(f"Generated Query String: {query_string}")
        
        # Process multiple queries if semicolon-separated
        queries = [q.strip() for q in query_string.split(';') if q.strip()]
        if not queries:
            queries = [query_string]  # Fallback to original string
        
        # Retrieve documents using multiple queries and combine results
        all_docs = []
        seen_content = set()
        
        for query in queries[:6]:  # Limit to top 6 queries
            try:
                docs = retriever.invoke(query.strip())
                for doc in docs:
                    # Avoid duplicate content
                    doc_content = doc.page_content[:100]  # Use first 100 chars as identifier
                    if doc_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc_content)
            except Exception as e:
                print(f"Error retrieving docs for query '{query}': {e}")
                continue
        
        # Limit total documents to avoid token overflow
        all_docs = all_docs[:20]
        print(f"Retrieved {len(all_docs)} unique documents from {len(queries)} queries")
        
        # Calculate trip duration and season information
        trip_context = ""
        if start_date and end_date:
            from datetime import datetime
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                duration = (end_dt - start_dt).days + 1
                
                # Determine season based on start date
                month = start_dt.month
                if month in [12, 1, 2]:
                    season = "winter"
                elif month in [3, 4, 5]:
                    season = "spring"
                elif month in [6, 7, 8]:
                    season = "summer"
                else:
                    season = "autumn"
                
                trip_context = f"""Trip Duration: {duration} days
                Travel Dates: {start_date} to {end_date}
                Season: {season}
                Month: {start_dt.strftime('%B')}"""
            except ValueError:
                trip_context = f"Travel Dates: {start_date} to {end_date}"
        
        # Enhanced prompt template with better document utilization
        prompt = PromptTemplate(
            input_variables=["user_input", "travel_plan", "docs", "city", "trip_context"],
            template="""Generate Egypt travel packing list based on:

USER: {user_input}
DESTINATION: {city}
TRIP: {trip_context}
PLAN: {travel_plan}
ITEMS: {docs}

Requirements:
- Cultural sensitivity (modest clothing for conservative areas)
- Climate appropriate ({trip_context})
- Activity-specific items from travel plan
- Egypt essentials (documents, adapters, currency)

Categories: Clothing/Modesty, Health/Safety, Electronics/Documents, Practical Items, Activity Essentials

Base recommendations on provided items database. Be comprehensive yet practical."""
        )
        
        prompt_text = prompt.format(
            user_input=user_input, 
            travel_plan=travel_plan, 
            docs=all_docs,
            city=city or "Egypt",
            trip_context=trip_context
        )
        
        packing_list = llm.invoke(prompt_text, config={"callbacks":[BaseCallbackHandler]})
        return packing_list
        
    

if __name__ == "__main__":
    start_time = time.time()
    
    vector_store_manager2 = VectorStoreManager()
    retriever2 = vector_store_manager2.get_retriever()
    
    # Test data
    city = "Luxor"
    favorite_places = "Cultural sites, historical landmarks, art galleries"
    visitor_type = "Foreign"
    start_date = "2025-07-15"
    end_date = "2025-07-17"
    budget = 3500
    
    # Calculate num_days for travel plan (which still expects num_days)
    from datetime import datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (end_dt - start_dt).days + 1
    
    llm_manager = LLMService(model_name="gemini-2.0-flash", provider='google-genai')
    travel_plan = llm_manager.travel_plan(retriever2, city, favorite_places, visitor_type, str(num_days), budget)
    
    print("\n📋 Generated Travel Plan:")
    print("=" * 50)
    print(json.dumps(travel_plan, indent=4))
    
    # Initialize packing list components
    packing_list_processor = PackingListProcessor("egy_guide_packing_list_final_100.csv")
    documents = packing_list_processor.df_to_documents()
    
    vector_store_manager1 = VectorStoreManager(documents=documents, path="packing_list")
    retriever = vector_store_manager1.get_retriever()
    
    llm_packing_list = PackingListGenerator(model_name="gemini-2.0-flash", provider='google-genai')
    
    # Test with enhanced parameters
    start_date = "2025-07-15"
    end_date = "2025-07-17"
    
    packing_list_result = llm_packing_list.generate_packing_list(
        retriever, travel_plan, city=city, start_date=start_date, end_date=end_date
    )
    
    # Print the packing list result in a more readable format
    print("\n🎒 Generated Packing List:")
    print("=" * 50)
    print(json.dumps(packing_list_result, indent=4))
    
    end_time = time.time()
    print(f"\n⏱️ Execution time: {end_time - start_time:.2f} seconds")
        