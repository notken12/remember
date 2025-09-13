#!/usr/bin/env python3

import os
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from VideoClip import VideoClip

# Load environment variables
load_dotenv()

class Question:
    """
    A class representing a question associated with a video clip for memory assistance.
    
    This class stores questions and answers related to video clips to help individuals
    with neurodegenerative diseases practice recall and memory exercises.
    """
    
    def __init__(self, video_clip: Optional[VideoClip] = None, text_cue: str = "", answer: str = ""):
        """
        Initialize a Question instance.
        
        Args:
            video_clip (VideoClip, optional): VideoClip instance associated with this question
            text_cue (str): Text cue/prompt for the question (default: "")
            answer (str): Answer to the question (default: "")
        """
        self.video_clip = video_clip
        self.text_cue = text_cue
        self.answer = answer
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
            
        self.supabase: Client = create_client(self.supabase_url, self.supabase_service_role_key)
    
    def push_to_supabase(self) -> bool:
        """
        Push the question data to Supabase database.
        
        This method inserts a new record into the questions table with:
        - video_id: UUID of the associated video clip
        - text_cue: The question/cue text
        - answer: The answer text
        
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If video_clip is None (video_id is required)
        """
        try:
            if self.video_clip is None:
                raise ValueError("VideoClip is required to push question to Supabase")
            
            print(f"📤 Pushing question to Supabase...")
            print(f"   Video ID: {self.video_clip.id}")
            print(f"   Text Cue: {self.text_cue[:50]}..." if len(self.text_cue) > 50 else f"   Text Cue: {self.text_cue}")
            print(f"   Answer: {self.answer[:50]}..." if len(self.answer) > 50 else f"   Answer: {self.answer}")
            
            # Insert the question record into the database
            response = self.supabase.table('test_questions').insert({
                'video_id': self.video_clip.id,
                'text_cue': self.text_cue,
                'answer': self.answer
            }).execute()
            
            if response.data:
                print(f"✅ Question pushed successfully to Supabase")
                print(f"   Record ID: {response.data[0].get('id', 'Unknown')}")
                return True
            else:
                print(f"❌ Failed to push question: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error pushing question to Supabase: {str(e)}")
            return False

def main():
    """
    Example usage of the Question class.
    """
    print("🤔 Question class example usage")
    print("-" * 50)
    
    try:
        # Example: Create a question with a video clip
        # Replace with actual video ID from your Supabase database
        video_id = "ade62cd7-3b6c-4e5e-a782-929dab2a2d16"
        
        # Create a VideoClip instance
        video_clip = VideoClip(video_id)
        
        # Create a Question instance
        question = Question(
            video_clip=video_clip,
            text_cue="What was the main activity happening in this video?",
            answer="A cat was playing with a toy in the living room"
        )
        
        print(f"Created question: {question}")
        
        # Push to Supabase
        success = question.push_to_supabase()
        
        if success:
            print("🎉 Question created and pushed successfully!")
        else:
            print("💥 Failed to push question to Supabase")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")
    
    print("-" * 50)
    print("Question class is ready to use!")
    print("Example usage:")
    print("  clip = VideoClip('your-video-uuid')")
    print("  question = Question(clip, 'What happened?', 'Something interesting')")
    print("  question.push_to_supabase()")


if __name__ == "__main__":
    main()
