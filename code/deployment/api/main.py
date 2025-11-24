from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import uuid
from PIL import Image
import logging
import time
import aiofiles
import asyncio

# Add code directory to path for imports
sys.path.insert(0, '/app')
from converter.latex_converter import LatexConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InstaTexScanner API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHARED_DATA_DIR = "/app/shared_data"
UPLOAD_DIR = os.path.join(SHARED_DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(SHARED_DATA_DIR, "outputs")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize LatexConverter (lazy loading to avoid import errors at startup)
latex_converter = None

def get_latex_converter():
    """Lazy initialization of LatexConverter"""
    global latex_converter
    if latex_converter is None:
        try:
            latex_converter = LatexConverter()
            logger.info("LatexConverter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LatexConverter: {e}")
            raise
    return latex_converter


@app.get("/")
async def root():
    return {"message": "InstaTexScanner API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/convert")
async def convert_image(
    file: UploadFile = File(...),
    output_format: str = "latex",
    segment_document: bool = True
):
    """
    Convert image to LaTeX, text, or PDF.
    
    Parameters:
    - file: Image file to process
    - output_format: "latex", "text", or "pdf"
    - segment_document: If True, segment document into text and images (default: True)
    """
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Please upload an image file")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    try:
        # Save uploaded file
        async with aiofiles.open(input_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Process based on format
        if output_format in ["latex", "text"]:
            # Use LatexConverter to extract text/LaTeX
            converter = get_latex_converter()
            
            # Run conversion in thread pool (CPU-intensive)
            def run_conversion():
                return converter.convert(
                    input_path, 
                    out_dir=OUTPUT_DIR,
                    segment_document=segment_document
                )
            
            result = await asyncio.get_event_loop().run_in_executor(
                executor, run_conversion
            )
            
            # Save text result
            if output_format == "text":
                output_filename = f"{file_id}_converted.txt"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(result["text"])
                
                return {
                    "success": True,
                    "message": "Text extracted successfully",
                    "text": result["text"],
                    "type": result["type"],
                    "download_url": f"/download/{output_filename}",
                    "filename": output_filename
                }
            else:  # latex
                # Return LaTeX content and file path
                latex_filename = os.path.basename(result["latex_file"])
                
                # Prepare response with image information
                response = {
                    "success": True,
                    "message": "LaTeX extracted successfully",
                    "latex": result["text"],
                    "type": result["type"],
                    "latex_file": result["latex_file"],
                    "download_url": f"/download/{latex_filename}",
                    "filename": latex_filename
                }
                
                # Add image information if available
                if "images" in result:
                    response["images"] = result["images"]
                    response["images_count"] = len(result["images"])
                    # Add download URLs for images
                    response["image_urls"] = [
                        f"/download/images/{os.path.basename(img_path)}" 
                        for img_path in result["images"]
                    ]
                
                if "segments_count" in result:
                    response["segments_count"] = result["segments_count"]
                
                return response

        elif output_format == "pdf":
            # Convert image to PDF
            output_filename = f"{file_id}_converted.pdf"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, "PDF", resolution=100.0)
            
            return {
                "success": True,
                "message": "Converted to PDF",
                "download_url": f"/download/{output_filename}",
                "filename": output_filename
            }
        else:
            raise HTTPException(400, "Invalid format. Use 'latex', 'text', or 'pdf'")
            
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Conversion failed: {str(e)}")
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/download/{filename:path}")
async def download_file(filename: str):
    """Download converted files and images"""
    # Handle nested paths (e.g., images/img_0.png)
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    # Security check: ensure path is within OUTPUT_DIR
    real_path = os.path.realpath(file_path)
    real_output_dir = os.path.realpath(OUTPUT_DIR)
    if not real_path.startswith(real_output_dir):
        raise HTTPException(403, "Access denied")

    logger.info(f"Looking for file: {file_path}")
    logger.info(f"File exists: {os.path.exists(file_path)}")

    if os.path.exists(file_path):
        # Determine media type
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            media_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'
        elif filename.endswith('.tex'):
            media_type = 'text/plain'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            filename=os.path.basename(filename),
            media_type=media_type
        )
    else:
        available_files = []
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
                available_files.append(rel_path)
        logger.warning(f"File not found. Available files: {available_files}")
        raise HTTPException(404, f"File not found. Available files: {available_files}")

@app.get("/files")
async def list_files():
    """List all converted files for debugging"""
    try:
        files = os.listdir(OUTPUT_DIR)
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)