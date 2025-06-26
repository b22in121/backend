from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import requests
from google import genai
from google.genai import types
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Gemini Vision & Video API", description="Simple FastAPI backend with Gemini for image and video analysis via URLs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Pydantic models for request bodies
class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    prompt: Optional[str] = "Analyze this image and describe what you see in detail."

class VideoAnalysisRequest(BaseModel):
    video_url: str  # Can be YouTube URL or direct video URL
    prompt: Optional[str] = "Analyze this video and describe what you see be detailed"
    start_offset: Optional[str] = None  # Format: "1250s" or "20m50s"
    end_offset: Optional[str] = None    # Format: "1570s" or "26m10s"
    fps: Optional[float] = None         # Custom frame rate (default is 1 FPS)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Gemini Vision & Video API is running!", 
        "status": "healthy", 
        "endpoints": {
            "image_analysis": "POST /analyze-image",
            "video_analysis": "POST /analyze-video"
        },
        "models": ["gemini-2.0-flash", "gemini-2.5-flash"],
        "supported_formats": {
            "images": ["JPEG", "PNG", "WebP", "HEIC", "HEIF"],
            "videos": ["MP4", "MPEG", "MOV", "AVI", "FLV", "MPG", "WebM", "WMV", "3GPP", "YouTube"]
        }
    }

@app.post("/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze an image from URL using Gemini Vision API
    """
    try:
        # Download image from URL
        response = requests.get(str(request.image_url), timeout=15)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL must point to an image")
        
        # Create image part using google.genai.types
        image_bytes = response.content
        image = types.Part.from_bytes(
            data=image_bytes, 
            mime_type=content_type
        )
        
        # Generate content using Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[request.prompt, image]
        )
        
        return JSONResponse(content={
            "success": True,
            "analysis": gemini_response.text,
            "image_url": str(request.image_url),
            "prompt_used": request.prompt,
            "model_used": "gemini-2.5-flash"
        })
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url.lower() for domain in youtube_domains)

def _create_video_part(video_url: str, start_offset: Optional[str] = None, 
                      end_offset: Optional[str] = None, fps: Optional[float] = None):
    """Create video part with optional metadata"""
    video_metadata = None
    
    if start_offset or end_offset or fps:
        metadata_kwargs = {}
        if start_offset:
            metadata_kwargs['start_offset'] = start_offset
        if end_offset:
            metadata_kwargs['end_offset'] = end_offset
        if fps:
            metadata_kwargs['fps'] = fps
        video_metadata = types.VideoMetadata(**metadata_kwargs)
    
    if _is_youtube_url(video_url):
        # For YouTube URLs, use file_data
        return types.Part(
            file_data=types.FileData(file_uri=video_url),
            video_metadata=video_metadata
        )
    else:
        # For direct video URLs, download and use inline_data
        try:
            response = requests.get(video_url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="URL must point to a video file")
            
            return types.Part(
                inline_data=types.Blob(
                    data=response.content,
                    mime_type=content_type
                ),
                video_metadata=video_metadata
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading video: {str(e)}")

@app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Analyze a video from URL (YouTube or direct video URL) using Gemini API
    """
    try:
        # Create video part with metadata
        video_part = _create_video_part(
            request.video_url, 
            request.start_offset, 
            request.end_offset, 
            request.fps
        )
        
        # Generate content using Gemini 2.0
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=types.Content(
                parts=[
                    video_part,
                    types.Part(text=request.prompt)
                ]
            )
        )
        
        return JSONResponse(content={
            "success": True,
            "analysis": gemini_response.text,
            "video_url": request.video_url,
            "prompt_used": request.prompt,
            "model_used": "gemini-2.0-flash",
            "processing_options": {
                "start_offset": request.start_offset,
                "end_offset": request.end_offset,
                "fps": request.fps
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
