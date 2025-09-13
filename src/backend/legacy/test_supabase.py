#!/usr/bin/env python3

import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def upload_video_to_supabase():
    """
    Upload cat.mp4 video to Supabase test_videos storage bucket and insert the video path into database.
    """
    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        # Video file path
        video_path = "../videos/cat.mp4"
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return False
        
        # Get file size for metadata
        file_size = os.path.getsize(video_path)
        
        print(f"Uploading video: {video_path}")
        print(f"File size: {file_size} bytes")
        
        # Upload file to Supabase Storage using the existing test_videos bucket
        with open(video_path, 'rb') as file:
            try:
                print("📤 Uploading video to test_videos bucket...")
                # Try with upsert option to overwrite if file exists
                storage_response = supabase.storage.from_('test_videos').upload(
                    path='cat.mp4',
                    file=file,
                    file_options={
                        "content-type": "video/mp4",
                    }
                )
                
                if hasattr(storage_response, 'path') and storage_response.path:
                    print(f"✅ Video uploaded successfully to storage: {storage_response.path}")
                    
                    # Get public URL for the uploaded video
                    public_url = supabase.storage.from_('test_videos').get_public_url('cat.mp4')
                    storage_path = storage_response.path
                    upload_success = True
                    
                else:
                    print("⚠️ Storage upload may have failed")
                    public_url = None
                    storage_path = None
                    upload_success = False
                    
            except Exception as storage_error:
                print(f"❌ Storage upload failed: {storage_error}")
                public_url = None
                storage_path = None
                upload_success = False
        
        if not upload_success:
            print("❌ Video upload to storage failed. Cannot proceed without storage upload.")
            return False
        
        # Insert record into database table
        # Using the available 'test_videos' table with 'video_path' column
        try:
            db_response = supabase.table('test_videos').insert({
                "video_path": storage_path
            }).execute()
            
            if db_response.data:
                print(f"✅ Database record created successfully:")
                print(f"   Record ID: {db_response.data[0]['id']}")
                print(f"   Video Path: {storage_path}")
                print(f"   Public URL: {public_url}")
                return True
            else:
                print(f"❌ Failed to create database record: {db_response}")
                return False
                
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            return False
                
    except Exception as e:
        print(f"❌ Error uploading video: {str(e)}")
        return False

def main():
    """
    Main function to run the video upload process.
    """
    print("🎬 Starting video upload to Supabase...")
    print("-" * 50)
    
    success = upload_video_to_supabase()
    
    print("-" * 50)
    if success:
        print("🎉 Video upload completed successfully!")
    else:
        print("💥 Video upload failed. Please check the errors above.")

if __name__ == "__main__":
    main()
