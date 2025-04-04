import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.callbacks import UsageMetadataCallbackHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TokenManager")

class TokenManager:
    def __init__(
        self,
        models_config: Optional[Dict] = None, # Configuration for models
        storage_path: str = "token_usage.json" # Path to store token usage data
    ):
        self.storage_path = Path(storage_path)
        self.callback_handler = UsageMetadataCallbackHandler()
        
        # Define default model configurations if not provided
        if models_config is None:
            self.models_config = {
                "high_priority": {
                    "name": "llama-3.3-70b-specdec",
                    "provider": "groq",
                    "token_limit": 96000,
                    "priority": 1
                },
                "medium_priority_1": {
                    "name": "llama-3.3-70b-versatile",
                    "provider": "groq",
                    "token_limit": 96000,
                    "priority": 2
                },
                "medium_priority_2": {
                  "name": "gemini-2.0-flash",
                  "provider": "google-genai",
                  "token_limit": 3500000,
                  "priority": 3  
                },
                "medium_priority_3": {
                  "name": "gemini-2.0-flash-lite",
                  "provider": "google-genai",
                  "token_limit": 3500000,
                  "priority": 4,
                },
                "low_priority_1": {
                    "name": "llama-3.2-90b-vision-preview",
                    "provider": "groq",
                    "token_limit": 246000,
                    "priority": 5
                },
                "low_priority_2": {
                    "name": "qwen-qwq-32b",
                    "provider": "groq",
                    "token_limit": 1000000,
                    "priority": 6
                },
                "low_priority_3": {
                    "name": "deepseek-r1-distill-llama-70b",
                    "provider": "groq",
                    "token_limit": 1000000,
                    "priority": 7
                },
                "low_priority_4": {
                    "name": 'gemini-2.5-pro-exp-03-25',
                    "provider": 'google-genai',
                    "token_limit": 100000,
                    "priority": 8
                },
                "low_priority_5": {
                    "name": "gemini-2.5-pro-preview-03-25",
                    "provider": 'google-genai',
                    "token_limit": 100000,
                    "priority": 9
                },
                "low_priority_6": {
                    "name": "meta/llama-3.3-70b-instruct",
                    "provider": "nvidia",
                    "token_limit": 1000000,
                    "priority": 10
                },
                # "fallback": {
                #     "name": "meta/llama-3.1-450b-instruct",
                #     "provider": "nvidia",
                #     "token_limit": 1000000,
                #     "priority": 7,
                # } 
            }
        else:
            self.models_config = models_config
            
        # Initialize token usage tracking
        self.usage_data = self._load_usage_data()
        
    def _get_today_date(self) -> str:
        """Get today's date as a string in YYYY-MM-DD format."""
        return datetime.datetime.now().strftime("%Y-%m-%d")
    
        
    def _get_highest_priority_model_id(self) -> str:
        """Get the model ID with the highest priority (lowest priority number)."""
        return min(self.models_config.items(), key=lambda x: x[1]["priority"])[0]
        
    def _load_usage_data(self) -> Dict:
        """Load token usage data from storage."""
        if not self.storage_path.exists():
            
            return {
                "last_updated": self._get_today_date(),
                "current_model_id": self._get_highest_priority_model_id(),
                "model_usage": {}
            }
            
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                
                
            # Check if we need to reset for a new day
            today = self._get_today_date()
            if data.get("last_updated") != today:
                data['last_updated'] = today
                # Reset current model to highest priority model for a new day
                data['current_model_id'] = self._get_highest_priority_model_id()
                
            return data
        except Exception as e:
            logger.error(f"Error loading token usage data: {e}. Using default values.")
            return {
                "last_updated": self._get_today_date(),
                "current_model_id": self._get_highest_priority_model_id(),
                "model_usage": {}
            }
            
            
    def _save_usage_data(self) -> None:
        """Save token usage data to storage."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving token usage data: {e}")
        
                
    def update_usage_from_callback(self) -> Dict[str, int]:
        """
        Update the usage data from callback handler.
        Returns a dictionary with token usage information.
        """
        today = self._get_today_date()
        
        # Get usage metadata from the callback handler
        usage_metadata = self.callback_handler.usage_metadata
        if not usage_metadata:
            return {"total_tokens": 0}
        
        
        # Calculate total tokens used in this request
        total_tokens = 0
        current_model_name = self.get_current_model()[0]
        current_model_usage = 0
        
        for model, stats in usage_metadata.items():
            # Track usage per model
            if model not in self.usage_data["model_usage"]:
                self.usage_data["model_usage"][model] = {}
            
            if today not in self.usage_data["model_usage"][model]:
                self.usage_data["model_usage"][model][today] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            
            # Update tokens for this model
            model_daily_usage = self.usage_data["model_usage"][model][today]
            model_daily_usage["input_tokens"] += stats.get("input_tokens", 0)
            model_daily_usage["output_tokens"] += stats.get("output_tokens", 0)
            model_daily_usage["total_tokens"] += stats.get("total_tokens", 0)
            
            # Update current model usage counter
            if model == current_model_name:
                current_model_usage = model_daily_usage["total_tokens"]
                
            # Add to total for all models
            total_tokens += stats.get("total_tokens", 0)
        
        # Check if we need to switch models
        self._check_and_switch_model()
        
        # Save updated data
        self._save_usage_data()
        
        # Reset the callback handler for the next request
        self.callback_handler = UsageMetadataCallbackHandler()
        
        return {
            "total_tokens": total_tokens,
            "current_model_usage": current_model_usage
        }
        
    def _check_and_switch_model(self) -> None:
        """
        Check token usage for each model and switch to a lower priority model
        if the current model exceeds its token limit.
        """
        today = self._get_today_date()
        current_model_id = self.usage_data["current_model_id"]
        current_model = self.models_config[current_model_id]
        current_model_name = current_model["name"]
        
        # # Skip check for NVIDIA provider models (unlimited tokens)
        # if current_model["provider"] == "nvidia":
        #     return
        
        # Get current model's usage
        current_usage = 0
        if current_model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][current_model_name]:
            current_usage = self.usage_data["model_usage"][current_model_name][today]["total_tokens"]
        
        # Check if current model exceeded its token limit
        if current_model["token_limit"] is not None and current_usage >= current_model["token_limit"]:
            # Find the next eligible model
            next_model_id = self._find_next_eligible_model(current_model_id)
            
            if next_model_id != current_model_id:
                self.usage_data["current_model_id"] = next_model_id
                next_model = self.models_config[next_model_id]
                logger.warning(
                    f"Model {current_model_name} exceeded its token limit ({current_usage}/{current_model['token_limit']}). "
                    f"Switching to {next_model['name']} (provider: {next_model['provider']})"
                )
            
    def _find_next_eligible_model(self, current_model_id: str) -> str:
        """
        Find the next eligible model based on priority.
        An eligible model is one that hasn't exceeded its token limit.
        If all models have exceeded their limits, return the highest priority model.
        """
        today = self._get_today_date()
        current_priority = self.models_config[current_model_id]["priority"]
        
        # Get all models sorted by priority (excluding the current one)
        eligible_models = sorted(
            [(model_id, config) for model_id, config in self.models_config.items() 
             if config["priority"] > current_priority],
            key=lambda x: x[1]["priority"]
        )
        
        # Check each model to see if it's under its token limit
        for model_id, config in eligible_models:
            # NVIDIA models are always eligible (no token limit)
            if config["provider"] == "nvidia":
                return model_id
            
            # Check token usage for non-NVIDIA models
            if config["token_limit"] is None:
                return model_id
                
            model_name = config["name"]
            current_usage = 0
            if model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][model_name]:
                current_usage = self.usage_data["model_usage"][model_name][today]["total_tokens"]
                
            if current_usage < config["token_limit"]:
                return model_id
        
        # If no eligible model found, stay with the current one
        return current_model_id
        
    def get_current_model(self) -> Tuple[str, str]:
        """
        Get the current model based on token usage.
        Returns a tuple of (model_name, provider)
        """
        model_id = self.usage_data["current_model_id"]
        model_config = self.models_config[model_id]
        return model_config["name"], model_config["provider"]
    
    def get_callback_handler(self) -> UsageMetadataCallbackHandler:
        """Get the callback handler for tracking token usage."""
        return self.callback_handler
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        today = self._get_today_date()
        current_model_id = self.usage_data["current_model_id"]
        current_model = self.models_config[current_model_id]
        
        # Calculate total usage across all models for today
        total_usage_today = 0
        for model_name, dates in self.usage_data["model_usage"].items():
            if today in dates:
                total_usage_today += dates[today]["total_tokens"]
        
        # Get usage for each model
        model_stats = {}
        for model_id, config in self.models_config.items():
            model_name = config["name"]
            usage = 0
            if model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][model_name]:
                usage = self.usage_data["model_usage"][model_name][today]["total_tokens"]
            
            limit = config["token_limit"]
            limit_str = "Unlimited" if limit is None else str(limit)
            
            model_stats[model_id] = {
                "name": model_name,
                "provider": config["provider"],
                "token_limit": limit_str,
                "usage_today": usage,
                "priority": config["priority"]
            }
        
        return {
            "date": today,
            "total_usage_today": total_usage_today,
            "current_model_id": current_model_id,
            "current_model": {
                "name": current_model["name"],
                "provider": current_model["provider"],
                "token_limit": "Unlimited" if current_model["token_limit"] is None else current_model["token_limit"]
            },
            "models": model_stats
        }        
            
              

        
        
        