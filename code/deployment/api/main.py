from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor
import os
import uuid
from PIL import Image
import logging
import time
import aiofiles

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


@app.get("/")
async def root():
    return {"message": "InstaTexScanner API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/convert")
async def convert_image(
    file: UploadFile = File(...),
    output_format: str = "pdf"
):
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Please upload an image file")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    # Determine output file extension
    if output_format == "pdf":
        output_filename = f"{file_id}_converted.pdf"
    elif output_format == "text":
        output_filename = f"{file_id}_converted.txt"
    else:
        raise HTTPException(400, "Invalid format. Use 'pdf' or 'text'")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    
    try:
        # Save uploaded file
        async with aiofiles.open(input_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Process based on format
        if output_format == "pdf":
            # Convert image to PDF
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

        elif output_format == "text":
            # Extract text
            # text = pytesseract.image_to_string(Image.open(input_path))
            
            # async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            #     await f.write(text)
            
            return {
                "success": True, 
                "message": "Extracted text",
                "download_url": f"/download/{file_id}_converted.txt"
            }
        else:
            raise HTTPException(400, "Invalid format. Use 'pdf' or 'text'")
            
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {str(e)}")
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download converted files"""
    file_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Looking for file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        available_files = os.listdir(OUTPUT_DIR)
        print(f"Available files: {available_files}")
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