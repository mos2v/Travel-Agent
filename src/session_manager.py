import uuid
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pickle


class SessionManager:
    """
    Manages travel plan sessions with in-memory cache and optional persistence.
    Sessions expire after a configurable time period.
    """
    
    def __init__(self, expiration_hours: int = 1, persist_to_file: bool = False):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.expiration_hours = expiration_hours
        self.persist_to_file = persist_to_file
        self.cache_file = Path("travel_plan_sessions.pkl")
        
        if self.persist_to_file and self.cache_file.exists():
            self._load_sessions_from_file()
    
    def create_session(self, travel_plan: dict, user_params: dict) -> str:
        """
        Create a new session with travel plan data.
        
        Args:
            travel_plan: The generated travel plan
            user_params: Original user parameters (city, favorite_places, etc.)
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = f"travel_session_{uuid.uuid4().hex[:12]}"
        expiry_time = datetime.now() + timedelta(hours=self.expiration_hours)
        
        session_data = {
            "travel_plan": travel_plan,
            "user_params": user_params,
            "created_at": datetime.now().isoformat(),
            "expires_at": expiry_time.isoformat(),
            "expiry_timestamp": expiry_time.timestamp()
        }
        
        self.sessions[session_id] = session_data
        
        if self.persist_to_file:
            self._save_sessions_to_file()
        
        print(f"✅ Created session {session_id} (expires: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by session ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session data if found and not expired, None otherwise
        """
        # Clean expired sessions first
        self._cleanup_expired_sessions()
        
        if session_id not in self.sessions:
            print(f"❌ Session {session_id} not found")
            return None
        
        session_data = self.sessions[session_id]
        
        # Double-check expiration
        if time.time() > session_data.get("expiry_timestamp", 0):
            print(f"⏰ Session {session_id} has expired")
            del self.sessions[session_id]
            if self.persist_to_file:
                self._save_sessions_to_file()
            return None
        
        print(f"✅ Retrieved session {session_id}")
        return session_data
    
    def extend_session(self, session_id: str, hours: int = None) -> bool:
        """
        Extend session expiration time.
        
        Args:
            session_id: The session identifier
            hours: Hours to extend (default uses original expiration_hours)
            
        Returns:
            True if session was extended, False if session not found
        """
        if session_id not in self.sessions:
            return False
        
        extend_hours = hours or self.expiration_hours
        new_expiry_time = datetime.now() + timedelta(hours=extend_hours)
        
        self.sessions[session_id]["expires_at"] = new_expiry_time.isoformat()
        self.sessions[session_id]["expiry_timestamp"] = new_expiry_time.timestamp()
        
        if self.persist_to_file:
            self._save_sessions_to_file()
        
        print(f"⏰ Extended session {session_id} until {new_expiry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Manually delete a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was deleted, False if session not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.persist_to_file:
                self._save_sessions_to_file()
            print(f"🗑️ Deleted session {session_id}")
            return True
        return False
    
    def get_active_sessions_count(self) -> int:
        """Get count of active (non-expired) sessions."""
        self._cleanup_expired_sessions()
        return len(self.sessions)
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions from memory."""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, data in self.sessions.items()
            if current_time > data.get("expiry_timestamp", 0)
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            print(f"🧹 Cleaned up expired session {session_id}")
        
        if expired_sessions and self.persist_to_file:
            self._save_sessions_to_file()
    
    def _save_sessions_to_file(self):
        """Save sessions to file for persistence."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.sessions, f)
        except Exception as e:
            print(f"⚠️ Failed to save sessions to file: {e}")
    
    def _load_sessions_from_file(self):
        """Load sessions from file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.sessions = pickle.load(f)
            print(f"📂 Loaded {len(self.sessions)} sessions from cache")
            # Clean up any expired sessions after loading
            self._cleanup_expired_sessions()
        except Exception as e:
            print(f"⚠️ Failed to load sessions from file: {e}")
            self.sessions = {}
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get metadata about a session without returning the full data.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session metadata
        """
        if session_id not in self.sessions:
            return {"exists": False}
        
        session_data = self.sessions[session_id]
        return {
            "exists": True,
            "created_at": session_data.get("created_at"),
            "expires_at": session_data.get("expires_at"),
            "user_params": session_data.get("user_params", {}),
            "is_expired": time.time() > session_data.get("expiry_timestamp", 0)
        }


# Convenience functions for common operations
def create_travel_session(travel_plan: dict, user_params: dict, session_manager: SessionManager) -> dict:
    """
    Create a travel session and return the response format.
    
    Args:
        travel_plan: The generated travel plan
        user_params: User parameters (city, favorite_places, etc.)
        session_manager: SessionManager instance
        
    Returns:
        Dictionary with travel_plan, session_id, and expires_at
    """
    session_id = session_manager.create_session(travel_plan, user_params)
    session_data = session_manager.get_session(session_id)
    
    return {
        "travel_plan": travel_plan,
        "session_id": session_id,
        "expires_at": session_data["expires_at"] if session_data else None,
        "notes": f"Session expires in {session_manager.expiration_hours} hour(s). Use session_id to generate packing list."
    }
