#!/usr/bin/env python3

import os
import uuid
from typing import Optional, List, TYPE_CHECKING
from supabase import create_client, Client
from dotenv import load_dotenv

# Import ChatMessage for type hints and functionality
if TYPE_CHECKING:
    from ChatMessage import ChatMessage

# Load environment variables
load_dotenv()

class ChatSession:
    """
    A class representing a chat session stored in Supabase.
    
    This class manages chat sessions with unique identifiers and provides
    Supabase integration for persistence.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a ChatSession instance.
        
        Args:
            session_id (str, optional): UUID of an existing session. If None, generates a new UUID.
        """
        self.id = session_id or str(uuid.uuid4())
        self.created_at = None  # Will be set by Supabase when inserted
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
            
        self.supabase: Client = create_client(self.supabase_url, self.supabase_service_role_key)
    
    def save_to_supabase(self) -> bool:
        """
        Save the chat session to Supabase database.
        
        This method inserts a new record into the chat_sessions table with:
        - id: UUID of the session
        - created_at: Timestamp (handled by Supabase)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"📤 Saving chat session to Supabase...")
            print(f"   Session ID: {self.id}")
            
            # Insert the session record into the database
            response = self.supabase.table('chat_sessions').insert({
                'id': self.id
            }).execute()
            
            if response.data:
                self.created_at = response.data[0].get('created_at')
                print(f"✅ Chat session saved successfully to Supabase")
                print(f"   Record ID: {self.id}")
                print(f"   Created at: {self.created_at}")
                return True
            else:
                print(f"❌ Failed to save chat session: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving chat session to Supabase: {str(e)}")
            return False
    
    def get_chat_messages(self) -> List['ChatMessage']:
        """
        Retrieve all chat messages for this session from Supabase database.
        
        This method runs a SQL query to fetch all messages associated with this session's UUID
        and returns them as a list of ChatMessage objects, ordered by creation time.
        
        Returns:
            List[ChatMessage]: List of ChatMessage objects for this session, ordered by creation time
        """
        try:
            # Import ChatMessage here to avoid circular imports
            from ChatMessage import ChatMessage
            
            print(f"📥 Loading chat messages for session from Supabase...")
            print(f"   Session ID: {self.id}")
            
            # Fetch all message records for this session from the database
            response = self.supabase.table('chat_messages').select('*').eq('session_id', self.id).order('created_at').execute()
            
            messages = []
            if response.data:
                for message_data in response.data:
                    message = ChatMessage(
                        content=message_data.get('content', ''),
                        session_id=message_data.get('session_id'),
                        role=message_data.get('role', 'user'),
                        message_id=message_data['id']
                    )
                    message.created_at = message_data.get('created_at')
                    messages.append(message)
                
                print(f"✅ Loaded {len(messages)} chat messages for session {self.id}")
            else:
                print(f"ℹ️ No chat messages found for session {self.id}")
            
            return messages
                
        except Exception as e:
            print(f"❌ Error loading chat messages for session from Supabase: {str(e)}")
            return []

def main():
    """
    Example usage of the ChatSession class.
    """
    print("💬 ChatSession class example usage")
    print("-" * 50)
    
    try:
        # Example 1: Create a new session
        print("Creating new chat session...")
        session = ChatSession()
        print(f"Created session: {session}")
        
        # Save to Supabase
        success = session.save_to_supabase()
        
        if success:
            print("🎉 Chat session created and saved successfully!")
            
            # Example 2: Get chat messages for this session
            print(f"\nGetting chat messages for session {session.id}...")
            messages = session.get_chat_messages()
            print(f"Found {len(messages)} messages for this session")
            
            for message in messages:
                print(f"  - {message.role}: {message.content[:50]}..." if len(message.content) > 50 else f"  - {message.role}: {message.content}")
            
        else:
            print("💥 Failed to save chat session to Supabase")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")
    
    print("-" * 50)
    print("ChatSession class is ready to use!")
    print("Example usage:")
    print("  session = ChatSession()")
    print("  session.save_to_supabase()")
    print("  messages = session.get_chat_messages()")

if __name__ == "__main__":
    main()