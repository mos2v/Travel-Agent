import json
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from langchain_core.callbacks import UsageMetadataCallbackHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TokenManager")

class TokenManager:
    def __init__(self, models_config: Optional[Dict] = None, storage_path: str = "token_usage.json"):
        self.storage_path = Path(storage_path)
        self.callback_handler = UsageMetadataCallbackHandler()
        
        # Default model configurations
        if models_config is None:
            self.models_config = {
                "high_priority_1": {
                    "name": "gemini-2.0-flash",
                    "provider": "google-genai",
                    "token_limit": 10000000,
                    "daily_request_limit": 1500,
                    "temperature": 0.0,
                    "priority": 1
                },
                "high_priority_2": {
                    "name": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "provider": "groq",
                    "token_limit": 500000,
                    "daily_request_limit": 1000,
                    "temperature": 0.0,
                    "priority": 2
                },
                "high_priority_3": {
                    "name": "gemini-2.5-flash",
                    "provider": "google-genai",
                    "token_limit": 2000000,
                    "daily_request_limit": 500,
                    "temperature": 0.2,  
                    "priority": 3
                },
                "medium_priority_1": {
                    "name": "gemini-2.5-flash-preview-04-17",
                    "provider": "google-genai",
                    "token_limit": 1000000,
                    "daily_request_limit": 500,  
                    "temperature": 0.0,  
                    "priority": 4
                },
                "medium_priority_2": {
                    "name": "gemini-2.5-flash-lite-preview-06-17",
                    "provider": "google-genai",
                    "token_limit": 4000000,
                    "daily_request_limit": 500, 
                    "temperature": 0.1,
                    "priority": 5  
                },
                "medium_priority_3": {
                    "name": "mistral-large-latest",
                    "provider": "mistralai",
                    "token_limit": 2000000,
                    "daily_request_limit": None, 
                    "temperature": 0.2,  
                    "priority": 6
                },
                "low_priority_1": {
                    "name": "gemini-2.0-flash-lite",
                    "provider": 'google-genai',
                    "token_limit": 5000000,
                    "daily_request_limit": 1500,  
                    "temperature": .3, 
                    "priority": 7
                },
                "low_priority_2": {
                    "name":"gemini-2.5-pro", 
                    "provider":"google-genai",
                    "token_limit": 500000,
                    "daily_request_limit": 25, 
                    "temperature": 0.1,  
                    "priority": 8
                },
                "low_priority_3": {
                    "name": "magistral-medium-2506",
                    "provider": "mistralai",
                    "token_limit": None,  
                    "daily_request_limit": None,  #
                    "temperature": 0,
                    "priority": 9
                },
                # "low_priority_4": {
                #     "name": "llama-3.3-70b-versatile",
                #     "provider": "groq",
                #     "token_limit": 96000,
                #     "daily_request_limit": None,
                #     "temperature": 0.0,
                #     "priority": 8
                # },
                # "low_priority_6": {
                #     "name": "llama-3.3-70b-specdec",
                #     "provider": "groq",
                #     "token_limit": 96000,
                #     "daily_request_limit": None,
                #     "temperature": 0.0,
                #     "priority": 9
                # },
                # "fallback": {
                #   "name": "meta/llama-3.3-70b-instruct",
                #     "provider": "nvidia",
                #     "token_limit": None,
                #     "daily_request_limit": None,
                #     "temperature": 0.2,
                #     "priority": 10,
                # } 
            }
        else:
            self.models_config = models_config
        
        self._normalize_model_configs()
        self.usage_data = self._load_usage_data()
        self._validate_and_fix_usage_data()
        
    def _get_today_date(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")
        
    def _get_highest_priority_model_id(self) -> str:
        return min(self.models_config.items(), key=lambda x: x[1]["priority"])[0]
        
    def _load_usage_data(self) -> Dict:
        if not self.storage_path.exists():
            return {
                "last_updated": self._get_today_date(),
                "current_model_id": self._get_highest_priority_model_id(),
                "model_usage": {},
                "request_counts": {}
            }
            
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                
            if "request_counts" not in data:
                data["request_counts"] = {}
                
            today = self._get_today_date()
            if data.get("last_updated") != today:
                data['last_updated'] = today
                data['current_model_id'] = self._get_highest_priority_model_id()
                data['request_counts'] = {}
                
            return data
        except Exception as e:
            logger.error(f"Error loading token usage data: {e}. Using defaults.")
            return {
                "last_updated": self._get_today_date(),
                "current_model_id": self._get_highest_priority_model_id(),
                "model_usage": {},
                "request_counts": {}
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
        current_model_name, current_provider = self.get_current_model()
        current_model_usage = 0
        
        # Update request count for the current model
        self._update_request_count(current_model_name)
        
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
    
    def _update_request_count(self, model_name: str) -> None:
        """
        Update request count for the specified model.
        """
        today = self._get_today_date()
        
        # Initialize request counts structure if needed
        if "request_counts" not in self.usage_data:
            self.usage_data["request_counts"] = {}
        
        if model_name not in self.usage_data["request_counts"]:
            self.usage_data["request_counts"][model_name] = {}
        
        if today not in self.usage_data["request_counts"][model_name]:
            self.usage_data["request_counts"][model_name][today] = 0
        
        # Increment request count
        self.usage_data["request_counts"][model_name][today] += 1
        
    def _check_and_switch_model(self) -> None:
        """
        Check token usage and request counts for each model and switch to a lower priority model
        if the current model exceeds its token limit or request limit.
        """
        today = self._get_today_date()
        current_model_id = self.usage_data["current_model_id"]
        
        # Add debugging
        logger.debug(f"🔍 Checking model switch for {current_model_id}")
        
        if current_model_id not in self.models_config:
            logger.error(f"❌ Current model ID {current_model_id} not found in models_config!")
            # Reset to highest priority model
            self.usage_data["current_model_id"] = self._get_highest_priority_model_id()
            self._save_usage_data()
            return
        
        current_model = self.models_config[current_model_id]
        current_model_name = current_model["name"]
        
        # Skip check for NVIDIA provider models (unlimited tokens and requests)
        if current_model["provider"] == "nvidia":
            return
        
        # Get current model's usage
        current_usage = 0
        if current_model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][current_model_name]:
            current_usage = self.usage_data["model_usage"][current_model_name][today]["total_tokens"]
        
        # Get current model's request count
        current_request_count = 0
        if (current_model_name in self.usage_data.get("request_counts", {}) and 
            today in self.usage_data["request_counts"][current_model_name]):
            current_request_count = self.usage_data["request_counts"][current_model_name][today]
        
        # Check if current model exceeded its token limit or request limit
        token_limit_exceeded = (current_model.get("token_limit") is not None and 
                               current_usage >= current_model.get("token_limit", float('inf')))
        
        request_limit_exceeded = (current_model.get("daily_request_limit") is not None and 
                                 current_request_count >= current_model.get("daily_request_limit", float('inf')))
        
        if token_limit_exceeded or request_limit_exceeded:
            # Find the next eligible model
            next_model_id = self._find_next_eligible_model(current_model_id)
            
            if next_model_id != current_model_id:
                self.usage_data["current_model_id"] = next_model_id
                next_model = self.models_config[next_model_id]
                
                if token_limit_exceeded:
                    logger.warning(
                        f"Model {current_model_name} exceeded its token limit ({current_usage}/{current_model['token_limit']}). "
                        f"Switching to {next_model['name']} (provider: {next_model['provider']})"
                    )
                else:  # request_limit_exceeded
                    logger.warning(
                        f"Model {current_model_name} exceeded its request limit ({current_request_count}/{current_model['daily_request_limit']}). "
                        f"Switching to {next_model['name']} (provider: {next_model['provider']})"
                    )
            
    def _find_next_eligible_model(self, current_model_id: str) -> str:
        """
        Find the next eligible model based on priority.
        An eligible model is one that hasn't exceeded its token limit or request limit.
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
        
        # Check each model to see if it's under its token limit and request limit
        for model_id, config in eligible_models:
            model_name = config["name"]
            
            # NVIDIA models are always eligible (no token limit, no request limit)
            if config["provider"] == "nvidia":
                return model_id
            
            # Check if model has unlimited tokens and requests
            if config.get("token_limit") is None and config.get("daily_request_limit") is None:
                return model_id
            
            # Check token usage
            token_usage = 0
            if model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][model_name]:
                token_usage = self.usage_data["model_usage"][model_name][today]["total_tokens"]
            
            # Check request count
            request_count = 0
            if (model_name in self.usage_data.get("request_counts", {}) and 
                today in self.usage_data["request_counts"][model_name]):
                request_count = self.usage_data["request_counts"][model_name][today]
            
            # Check if model is under both token and request limits
            token_limit_ok = config.get("token_limit") is None or token_usage < config.get("token_limit", float('inf'))
            request_limit_ok = config.get("daily_request_limit") is None or request_count < config.get("daily_request_limit", float('inf'))
            
            if token_limit_ok and request_limit_ok:
                return model_id
        
        # If no eligible model found, stay with the current one
        return current_model_id
        
    def get_current_model(self) -> Tuple[str, str]:
        """
        Get the current model based on token usage.
        Returns a tuple of (model_name, provider)
        """
        model_id = self.usage_data["current_model_id"]
        
        # Safety check: ensure model_id exists in models_config
        if model_id not in self.models_config:
            logger.warning(f"Current model ID {model_id} not found in config. Resetting to default.")
            model_id = self._get_highest_priority_model_id()
            self.usage_data["current_model_id"] = model_id
            self._save_usage_data()
        
        model_config = self.models_config[model_id]
        return model_config["name"], model_config["provider"]
    
    def get_model_temperature(self, model_name: Optional[str] = None) -> float:
        """
        Get the temperature setting for the specified model.
        If no model is specified, returns the temperature for the current model.
        """
        if model_name is None:
            model_id = self.usage_data["current_model_id"]
            return self.models_config[model_id].get("temperature", 0.0)
        
        # Find the model by name
        for model_id, config in self.models_config.items():
            if config["name"] == model_name:
                return config.get("temperature", 0.0)
                
        # Default temperature if model not found
        return 0.0
    
    def set_model_temperature(self, model_name: str, temperature: float) -> bool:
        """
        Set the temperature for the specified model.
        Returns True if successful, False if model not found.
        """
        # Ensure temperature is within valid range
        temperature = max(0.0, min(1.0, temperature))
        
        # Find the model by name and update its temperature
        for model_id, config in self.models_config.items():
            if config["name"] == model_name:
                self.models_config[model_id]["temperature"] = temperature
                self._save_usage_data()
                logger.info(f"Temperature for model {model_name} set to {temperature}")
                return True
                
        logger.warning(f"Model {model_name} not found, temperature not updated")
        return False
    
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
            
            # Get token usage
            token_usage = 0
            if model_name in self.usage_data["model_usage"] and today in self.usage_data["model_usage"][model_name]:
                token_usage = self.usage_data["model_usage"][model_name][today]["total_tokens"]
            
            # Get request count
            request_count = 0
            if (model_name in self.usage_data.get("request_counts", {}) and 
                today in self.usage_data["request_counts"][model_name]):
                request_count = self.usage_data["request_counts"][model_name][today]
            
            token_limit = config["token_limit"]
            token_limit_str = "Unlimited" if token_limit is None else str(token_limit)
            
            request_limit = config.get("daily_request_limit")
            request_limit_str = "Unlimited" if request_limit is None else str(request_limit)
            
            model_stats[model_id] = {
                "name": model_name,
                "provider": config["provider"],
                "token_limit": token_limit_str,
                "usage_today": token_usage,
                "daily_request_limit": request_limit_str,
                "requests_today": request_count,
                "temperature": config.get("temperature", 0.0),
                "priority": config["priority"]
            }
        
        return {
            "date": today,
            "total_usage_today": total_usage_today,
            "current_model_id": current_model_id,
            "current_model": {
                "name": current_model["name"],
                "provider": current_model["provider"],
                "token_limit": "Unlimited" if current_model["token_limit"] is None else current_model["token_limit"],
                "daily_request_limit": "Unlimited" if current_model.get("daily_request_limit") is None else current_model["daily_request_limit"],
                "temperature": current_model.get("temperature", 0.0)
            },
            "models": model_stats
        }
    
    def set_current_model(self, model_name: str, provider: str) -> bool:
        """
        Manually set the current model and provider.
        Returns True if successful, False if model not found or invalid.
        """
        # First, check if this exact model and provider combination exists
        for model_id, config in self.models_config.items():
            if config["name"] == model_name and config["provider"] == provider:
                self.usage_data["current_model_id"] = model_id
                self._save_usage_data()
                logger.info(f"Current model manually set to {model_name} (provider: {provider})")
                return True
        
        # If not found, try to add it as a new model configuration
        # Generate a new model ID
        new_model_id = f"custom_{len(self.models_config)}"
        
        # Add new model with default settings
        self.models_config[new_model_id] = {
            "name": model_name,
            "provider": provider,
            "token_limit": 1000000,  # Default limit
            "daily_request_limit": 1000,  # Default limit
            "temperature": 0.0,  # Default temperature
            "priority": 999  # Low priority for custom models
        }
        
        # Set as current model
        self.usage_data["current_model_id"] = new_model_id
        self._save_usage_data()
        
        logger.info(f"Added new model {model_name} (provider: {provider}) and set as current")
        return True
    
    def _normalize_model_configs(self) -> None:
        """Ensure all model configurations have required fields with default values."""
        required_fields = {
            "name": "",
            "provider": "",
            "token_limit": 1000000,
            "daily_request_limit": 1000,
            "temperature": 0.0,
            "priority": 999
        }
        
        for model_id, config in self.models_config.items():
            for field, default_value in required_fields.items():
                if field not in config:
                    config[field] = default_value
                    logger.warning(f"Added missing field '{field}' to model {model_id} with default value {default_value}")
    
    def _validate_and_fix_usage_data(self) -> None:
        """Validate and fix inconsistencies in usage data."""
        # Ensure current_model_id exists in models_config
        current_model_id = self.usage_data.get("current_model_id")
        if not current_model_id or current_model_id not in self.models_config:
            logger.warning(f"Invalid current_model_id: {current_model_id}. Resetting to highest priority model.")
            self.usage_data["current_model_id"] = self._get_highest_priority_model_id()
            self._save_usage_data()
        
        # Ensure request_counts exists
        if "request_counts" not in self.usage_data:
            self.usage_data["request_counts"] = {}
            
        # Ensure model_usage exists
        if "model_usage" not in self.usage_data:
            self.usage_data["model_usage"] = {}