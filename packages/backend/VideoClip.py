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
            response = self.supabase.table('videos').select('video_path').eq('id', self.id).single().execute()
            
            if not response.data:
                raise Exception(f"Video with ID {self.id} not found in database")
                
            video_path = response.data['video_path']
            
            # Download the video file from storage
            storage_response = self.supabase.storage.from_('videos').download(video_path)
            
            if not storage_response:
                raise Exception(f"Failed to download video from storage path: {video_path}")
                
            return storage_response
            
        except Exception as e:
            raise Exception(f"Error fetching video from Supabase: {str(e)}")
    
    def get_video_path(self) -> str:
        """
        Return the storage path for this video's object in Supabase.
        """
        print(f"🔎 [VideoClip:{self.id}] Fetching video_path from Supabase...")
        response = self.supabase.table('videos').select('video_path').eq('id', self.id).single().execute()
        if not response.data or 'video_path' not in response.data:
            raise Exception(f"Video with ID {self.id} not found in database")
        path = response.data['video_path']
        print(f"✅ [VideoClip:{self.id}] video_path: {path}")
        return path

    def download_to_tempfile(self, suffix: str = '.mp4') -> str:
        """
        Download the video to a temporary file and return the local file path.
        Caller is responsible for deleting the returned file when done.
        """
        video_path = self.get_video_path()
        print(f"📥 [VideoClip:{self.id}] Downloading from storage path: {video_path}")
        storage_response = self.supabase.storage.from_('videos').download(video_path)
        if not storage_response:
            raise Exception(f"Failed to download video from storage path: {video_path}")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(storage_response)
            print(f"💾 [VideoClip:{self.id}] Saved temp file: {temp_file.name}")
            return temp_file.name

    def upload_to_gemini(self, local_path: str):
        """
        Upload a local video file to Gemini and wait until processing completes.
        Returns the uploaded file handle usable as a multimodal part.
        """
        print(f"🚀 [VideoClip:{self.id}] Uploading to Gemini: {local_path}")
        video_file = genai.upload_file(path=local_path, mime_type="video/mp4")
        import time
        while getattr(video_file, 'state', None) and getattr(video_file.state, 'name', '') == "PROCESSING":
            print("⏳ [VideoClip] Gemini processing...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        if getattr(video_file, 'state', None) and getattr(video_file.state, 'name', '') == "FAILED":
            raise Exception("Video processing failed")
        try:
            uri = getattr(video_file, 'uri', None) or getattr(video_file, 'name', None)
            print(f"✅ [VideoClip:{self.id}] Gemini upload ready: {uri}")
        except Exception:
            pass
        return video_file

    def make_langchain_media_part(self) -> tuple:
        """
        Produce a LangChain-compatible media content part for this video.

        Returns a tuple: (media_part_dict, temp_path)
        Caller is responsible for deleting temp_path when finished.
        """
        temp_path = self.download_to_tempfile()
        with open(temp_path, 'rb') as f:
            data = f.read()
        media_part = {
            "type": "media",
            "mime_type": "video/mp4",
            "data": data,
        }
        return media_part, temp_path
    
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
                # Upload the video file to Gemini (reusing upload utility)
                video_file = self.upload_to_gemini(temp_video_path)

                model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
                
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
            # You may need to add an 'annotation' column to your videos table
            
            response = self.supabase.table('videos').update({
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
            
            # Step 1: Download video to a temp file
            print("📥 Downloading video to a temporary file...")
            temp_video_path = self.download_to_tempfile()
            print(f"✅ Video downloaded successfully to {temp_video_path}")
            
            # Step 2: Generate annotation with Gemini
            print("🤖 Generating annotation with Gemini 2.5 Flash Lite...")
            try:
                # Read bytes to reuse existing annotation generator
                with open(temp_video_path, 'rb') as f:
                    video_bytes = f.read()
                self.annotation = self._generate_annotation_with_gemini(video_bytes)
            finally:
                try:
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                except Exception:
                    pass
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
    
def main():
    """
    Example usage of the VideoClip class.
    """
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")

        supabase: Client = create_client(supabase_url, supabase_service_role_key)

        # Fetch all video IDs from the database
        print("🔎 Fetching video IDs from Supabase...")
        response = supabase.table('videos').select('id').execute()
        rows = response.data or []

        if not rows:
            print("No videos found in 'videos' table.")
            return

        print(f"Found {len(rows)} videos. Beginning annotation...")

        # Iterate through each video and annotate
        for row in rows:
            video_id = row.get('id')
            if not video_id:
                continue

            print(f"\n=== Processing video {video_id} ===")
            try:
                clip = VideoClip(video_id)
                success = clip.annotate()
                if success:
                    print(f"✅ Finished annotating video {video_id}")
                else:
                    print(f"❌ Failed to annotate video {video_id}")
            except Exception as e:
                print(f"❌ Error processing video {video_id}: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
