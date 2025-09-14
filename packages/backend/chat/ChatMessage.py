#!/usr/bin/env python3

import os
import uuid
from typing import Optional, List, Any, Dict
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChatMessage:
    """
    A class representing a chat message stored in Supabase.
    
    This class manages individual chat messages with content, role, and session association,
    providing Supabase integration for persistence.
    """
    
    def __init__(self, content: str = "", session_id: Optional[str] = None, role: str = "user", message_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Initialize a ChatMessage instance.
        
        Args:
            content (str): The content/text of the message
            session_id (str, optional): UUID of the associated chat session
            role (str): Role of the message sender (default: "user")
            message_id (str, optional): UUID of an existing message. If None, generates a new UUID.
        """
        self.id = message_id or str(uuid.uuid4())
        self.content = content
        self.session_id = session_id
        self.role = role
        self.created_at = None  # Will be set by Supabase when inserted
        self.data = data
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
            
        self.supabase: Client = create_client(self.supabase_url, self.supabase_service_role_key)
    
    def save_to_supabase(self) -> bool:
        """
        Save the chat message to Supabase database.
        
        This method inserts a new record into the chat_messages table with:
        - id: UUID of the message
        - content: Message text content
        - session_id: UUID of the associated session
        - role: Role of the message sender
        - created_at: Timestamp (handled by Supabase)
        
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If session_id is None (required for message association)
        """
        try:
            if self.session_id is None:
                raise ValueError("session_id is required to save message to Supabase")
            
            print(f"📤 Saving chat message to Supabase...")
            print(f"   Message ID: {self.id}")
            print(f"   Session ID: {self.session_id}")
            print(f"   Role: {self.role}")
            print(f"   Content: {self.content[:50]}..." if len(self.content) > 50 else f"   Content: {self.content}")
            
            # Insert the message record into the database
            payload = {
                'id': self.id,
                'content': self.content,
                'session_id': self.session_id,
                'role': self.role
            }
            # Include structured message data if provided (for LangGraph state reconstruction)
            if self.data is not None:
                payload['data'] = self.data

            response = self.supabase.table('chat_messages').insert(payload).execute()
            
            if response.data:
                self.created_at = response.data[0].get('created_at')
                print(f"✅ Chat message saved successfully to Supabase")
                print(f"   Record ID: {self.id}")
                print(f"   Created at: {self.created_at}")
                return True
            else:
                print(f"❌ Failed to save chat message: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving chat message to Supabase: {str(e)}")
            return False
    
def main():
    """
    Example usage of the ChatMessage class.
    """
    print("💬 ChatMessage class example usage")
    print("-" * 50)
    
    try:
        # Example: Create a message with a session ID
        # Replace with actual session ID from your Supabase database
        session_id = "550e8400-e29b-41d4-a716-446655440000"  # Example UUID
        
        # Example 1: Create a new user message
        print("Creating new chat message...")
        message = ChatMessage(
            content="Hello, this is a test message!",
            session_id=session_id,
            role="user"
        )
        print(f"Created message: {message}")
        
        # Save to Supabase
        success = message.save_to_supabase()
        
        if success:
            print("🎉 Chat message created and saved successfully!")
            
            # Example 2: Get all messages for the session
            print(f"\nGetting all messages for session {session_id}...")
            session_messages = ChatMessage.get_messages_by_session(session_id)
            print(f"Found {len(session_messages)} messages for session")
            
        else:
            print("💥 Failed to save chat message to Supabase")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")
    
    print("-" * 50)
    print("ChatMessage class is ready to use!")
    print("Example usage:")
    print("  message = ChatMessage('Hello!', session_id='your-session-uuid', role='user')")
    print("  message.save_to_supabase()")
    print("  messages = ChatMessage.get_messages_by_session('session-uuid')")

if __name__ == "__main__":
    main()
