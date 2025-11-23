#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for local launch of the InstaTexScanner frontend server
"""
import http.server
import socketserver
import os
from pathlib import Path
import webbrowser
import threading
import time

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
FRONTEND_DIR = PROJECT_ROOT / "code" / "deployment" / "app"

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Improved logging
        print(f"[Frontend] {format % args}")

def open_browser():
    """Opens browser 1 second after server start"""
    time.sleep(1)
    url = f"http://localhost:{PORT}"
    print(f"🌐 Opening browser: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    # Check if frontend directory exists
    if not FRONTEND_DIR.exists():
        print(f"❌ Error: Frontend directory not found: {FRONTEND_DIR}")
        exit(1)
    
    # Check for required files
    required_files = ["index.html", "script.js"]
    for file in required_files:
        if not (FRONTEND_DIR / file).exists():
            print(f"⚠️  Warning: File {file} not found in {FRONTEND_DIR}")
    
    # Start server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print("=" * 60)
        print("🚀 Starting InstaTexScanner Frontend Server")
        print("=" * 60)
        print(f"📁 Directory: {FRONTEND_DIR}")
        print(f"🌐 Server available at: http://localhost:{PORT}")
        print(f"📄 Open in browser: http://localhost:{PORT}")
        print("\nPress Ctrl+C to stop server\n")
        print("=" * 60)
        
        # Open browser in a separate thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping server...")
            httpd.shutdown()
            print("✅ Server stopped")

