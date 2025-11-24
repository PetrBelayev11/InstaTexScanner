#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for local launch of InstaTexScanner API server
"""
import os
import sys
from pathlib import Path

# IMPORTANT: First load the standard 'code' module into sys.modules,
# to avoid conflict with the project's 'code' folder
# This must be done BEFORE adding the 'code' folder to sys.path
import importlib
_std_code_module = importlib.import_module('code')
# Ensure that the standard module is registered in sys.modules
sys.modules['code'] = _std_code_module

# Add project root directory to path
project_root = Path(__file__).parent.absolute()
code_dir = project_root / "code"

# Add code_dir to path (standard module code is already in sys.modules)
sys.path.insert(0, str(code_dir))

# Change working directory to project root
os.chdir(project_root)

# Import and run application
if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    shared_data_dir = project_root / "shared_data"
    upload_dir = shared_data_dir / "uploads"
    output_dir = shared_data_dir / "outputs"
    
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change paths in main.py for local launch
    # Temporarily change environment variables
    os.environ["SHARED_DATA_DIR"] = str(shared_data_dir)
    

    print("🚀 Starting InstaTexScanner API server...")
    print(f"📁 Working directory: {project_root}")
    print("🌐 API available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop server\n")
    
    uvicorn.run(
        "deployment.api.main_local:app",  # Import string format for reload
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "code")],
        log_level="info"
    )
