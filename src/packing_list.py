import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.travel_plan_v3 import VectorStoreManager, LLMService
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
class processPackingList:
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
    

class LLMPackingList:
    def __init__(self, model_name, provider='google-genai', temperature: float = 0.3):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        
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
        Generate a query based on user input, travel plan, and trip details.
        """
        llm = self.initialize_llm()
        
        # Build context with available information
        context_parts = [f"User input: {user_input}", f"Travel plan: {travel_plan}"]
        
        if city:
            context_parts.append(f"Destination city: {city}")
        if start_date and end_date:
            context_parts.append(f"Trip dates: from {start_date} to {end_date}")
        elif start_date:
            context_parts.append(f"Trip start date: {start_date}")
        elif end_date:
            context_parts.append(f"Trip end date: {end_date}")
        
        context = "\n".join(context_parts)
        
        prompt = PromptTemplate(
            input_variables=["context"],
            template="""Based on the following travel information, generate a focused search query to find relevant packing list items:
            
            {context}

            Generate a concise search query that will help retrieve the most relevant packing items for this specific trip. Consider:
            - The destination and its climate
            - The travel dates and season
            - Activities mentioned in the travel plan
            - Duration of the trip
            - Any specific requirements mentioned by the user

            Query:"""
        )
        prompt_text = prompt.format(context=context)
        query = llm.invoke(prompt_text)
        return query.content

    def generate_packing_list(self, retriever, travel_plan, city: str = None, start_date: str = None, end_date: str = None):
        """
        Generate a comprehensive packing list based on travel plan and trip details.
        """
        llm = self.initialize_llm()
        user_input = f"I'm traveling to {city} for {len(travel_plan['days'])} days. What should I pack?"
        # Generate enhanced query with all available information
        query = self.generate_query_based_on_user_input(user_input, travel_plan, city, start_date, end_date)
        print(f"Generated Query: {query}")
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
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
        
        # Enhanced prompt template
        prompt = PromptTemplate(
            input_variables=["user_input", "travel_plan", "docs", "city", "trip_context"],
            template="""You are an expert travel packing advisor. Generate a comprehensive and practical packing list based on the following information:
            
            USER REQUEST: {user_input}

            DESTINATION: {city}

            TRIP DETAILS: {trip_context}

            TRAVEL PLAN: {travel_plan}

            RELEVANT PACKING INFORMATION: {docs}

            Create a detailed packing list that includes:

            1. **CLOTHING & ACCESSORIES**
            - Weather-appropriate clothing for the destination and season
            - Activity-specific clothing based on the travel plan
            - Footwear recommendations
            - Accessories and sun protection

            2. **PERSONAL CARE & HEALTH**
            - Essential toiletries and medications
            - Health and safety items specific to the destination
            - First aid basics

            3. **ELECTRONICS & DOCUMENTS**
            - Travel documents and copies
            - Electronics and chargers
            - Photography equipment if relevant

            4. **ACTIVITY-SPECIFIC ITEMS**
            - Items needed for specific activities mentioned in the travel plan
            - Cultural considerations for the destination

            5. **PRACTICAL ITEMS**
            - Luggage recommendations
            - Money and payment considerations
            - Useful travel accessories

            For each category, provide specific recommendations based on:
            - The destination's climate and culture
            - The specific activities planned
            - The duration of the trip
            - The season/time of year

            Make the list practical and avoid over-packing while ensuring nothing essential is missed.
            Format the response in a clear, organized manner with categories and bullet points."""
)
        
        prompt_text = prompt.format(
            user_input=user_input, 
            travel_plan=travel_plan, 
            docs=docs,
            city=city or "the destination",
            trip_context=trip_context
        )
        
        packing_list = llm.invoke(prompt_text)
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
    packing_list_processor = processPackingList("egy_guide_packing_list_final_100.csv")
    documents = packing_list_processor.df_to_documents()
    
    vector_store_manager1 = VectorStoreManager(documents=documents, path="packing_list")
    retriever = vector_store_manager1.get_retriever()
    
    llm_packing_list = LLMPackingList(model_name="gemini-2.0-flash", provider='google-genai')
    
    # Test with enhanced parameters
    start_date = "2025-07-15"
    end_date = "2025-07-17"
    
    packing_list_result = llm_packing_list.generate_packing_list(
        retriever, travel_plan, city=city, start_date=start_date, end_date=end_date
    )
    
    # Print the packing list result in a more readable format
    print("\n🎒 Generated Packing List:")
    print("=" * 50)
    print(packing_list_result.content)
    
    end_time = time.time()
    print(f"\n⏱️ Execution time: {end_time - start_time:.2f} seconds")
        