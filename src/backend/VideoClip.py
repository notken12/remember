#!/usr/bin/env python3

import os
import tempfile
from uuid import UUID
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VideoClip:
    """
    A class representing a video clip stored in Supabase with annotation capabilities.
    
    This class is designed for videos captured from smart glasses to help individuals 
    with neurodegenerative diseases remember their experiences by providing detailed 
    contextual annotations.
    """
    
    def __init__(self, video_id: str):
        """
        Initialize a VideoClip instance.
        
        Args:
            video_id (str): UUID of the video in Supabase database
        """
        self.id = video_id
        self.annotation = ""
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_service_role_key)
        
        # Initialize Gemini client
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=gemini_api_key)
        
    def _fetch_video_from_supabase(self) -> bytes:
        """
        Fetch video data from Supabase storage.
        
        Returns:
            bytes: Video file data
            
        Raises:
            Exception: If video cannot be found or downloaded
        """
        try:
            # First, get the video record from the database to get the video_path
            response = self.supabase.table('test_videos').select('video_path').eq('id', self.id).single().execute()
            
            if not response.data:
                raise Exception(f"Video with ID {self.id} not found in database")
                
            video_path = response.data['video_path']
            
            # Download the video file from storage
            storage_response = self.supabase.storage.from_('test_videos').download(video_path)
            
            if not storage_response:
                raise Exception(f"Failed to download video from storage path: {video_path}")
                
            return storage_response
            
        except Exception as e:
            raise Exception(f"Error fetching video from Supabase: {str(e)}")
    
    def _generate_annotation_with_gemini(self, video_data: bytes) -> str:
        """
        Generate annotation for the video using Gemini 2.0 Flash.
        
        Args:
            video_data (bytes): Video file data
            
        Returns:
            str: Generated annotation
            
        Raises:
            Exception: If annotation generation fails
        """
        try:
            # Create a temporary file to store the video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_data)
                temp_video_path = temp_file.name
            
            try:
                # Upload the video file to Gemini
                video_file = genai.upload_file(path=temp_video_path, mime_type="video/mp4")
                
                # Wait for the file to be processed
                import time
                while video_file.state.name == "PROCESSING":
                    print("Processing video...")
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    raise Exception("Video processing failed")
                
                model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                
                # Craft a concise but detailed prompt for memory assistance
                prompt = """
                Create a brief but vivid annotation of this smart glasses video for memory recall assistance. Focus on sensory details and atmosphere in 2-3 short paragraphs maximum.
                
                Include: Visual details (people, objects, colors, lighting), audio (music, voices, sounds), emotional atmosphere, and key interactions. Be specific about memorable elements that serve as memory anchors.
                
                Write in present tense, no introductory phrases, no formatting headers. Start directly with the scene description.
                """
                
                # Generate the annotation
                response = model.generate_content([video_file, prompt])
                
                # Clean up the temporary file
                os.unlink(temp_video_path)
                
                return response.text
                
            finally:
                # Ensure temporary file is cleaned up even if an error occurs
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    
        except Exception as e:
            raise Exception(f"Error generating annotation with Gemini: {str(e)}")
    
    def _update_annotation_in_supabase(self, annotation: str) -> bool:
        """
        Update the annotation in Supabase database.
        
        Args:
            annotation (str): The annotation to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if annotation column exists, if not we need to add it
            # For now, we'll assume the table structure supports annotations
            # You may need to add an 'annotation' column to your test_videos table
            
            response = self.supabase.table('test_videos').update({
                'annotation': annotation
            }).eq('id', self.id).execute()
            
            if response.data:
                print(f"✅ Annotation updated successfully for video {self.id}")
                return True
            else:
                print(f"❌ Failed to update annotation: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating annotation in Supabase: {str(e)}")
            return False
    
    def annotate(self) -> bool:
        """
        Main method to annotate the video.
        
        This method:
        1. Fetches the video from Supabase storage
        2. Sends it to Gemini 2.0 Flash for annotation
        3. Updates the annotation in Supabase database
        
        Returns:
            bool: True if annotation was successful, False otherwise
        """
        try:
            print(f"🎬 Starting annotation process for video {self.id}...")
            
            # Step 1: Fetch video from Supabase
            print("📥 Fetching video from Supabase...")
            video_data = self._fetch_video_from_supabase()
            print(f"✅ Video fetched successfully ({len(video_data)} bytes)")
            
            # Step 2: Generate annotation with Gemini
            print("🤖 Generating annotation with Gemini 2.0 Flash...")
            self.annotation = self._generate_annotation_with_gemini(video_data)
            print("✅ Annotation generated successfully")
            
            # Step 3: Update annotation in Supabase
            print("📤 Updating annotation in Supabase...")
            success = self._update_annotation_in_supabase(self.annotation)
            
            if success:
                print("🎉 Video annotation completed successfully!")
                return True
            else:
                print("💥 Failed to update annotation in database")
                return False
                
        except Exception as e:
            print(f"❌ Error during annotation process: {str(e)}")
            return False
    
    def get_annotation(self) -> str:
        """
        Get the current annotation for this video.
        
        Returns:
            str: The annotation text
        """
        return self.annotation
    
    def __str__(self) -> str:
        """String representation of the VideoClip."""
        return f"VideoClip(id={self.id}, annotation_length={len(self.annotation)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the VideoClip."""
        return f"VideoClip(id='{self.id}', annotation='{self.annotation[:50]}...' if len(self.annotation) > 50 else annotation='{self.annotation}')"


def main():
    """
    Example usage of the VideoClip class.
    """
    # Example usage - replace with actual video ID from your Supabase database
    video_id = "ade62cd7-3b6c-4e5e-a782-929dab2a2d16"
    # 
    try:
        clip = VideoClip(video_id)
        success = clip.annotate()
        
        if success:
            print(f"Annotation: {clip.get_annotation()}")
        else:
            print("Failed to annotate video")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("VideoClip class is ready to use!")
    print("Example usage:")
    print("  clip = VideoClip('your-video-uuid')")
    print("  clip.annotate()")
    print("  print(clip.get_annotation())")


if __name__ == "__main__":
    main()
