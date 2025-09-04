#!/usr/bin/env python3
"""
Startup script for the Document Processing Feedback UI

This script starts both the backend API and the frontend Streamlit application
for the human-in-the-loop feedback system.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    backend_process = subprocess.Popen([
        sys.executable, "run_api.py"
    ], cwd=Path(__file__).parent)
    return backend_process

def start_frontend():
    """Start the Streamlit frontend application."""
    print("🎨 Starting Streamlit frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "doc_processor/frontend/app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ], cwd=Path(__file__).parent)
    return frontend_process

def main():
    """Main function to start both services."""
    print("=" * 60)
    print("📄 Document Processing Feedback UI")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not Path("run_api.py").exists():
        print("❌ Error: run_api.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    if not Path("doc_processor/frontend/app.py").exists():
        print("❌ Error: Frontend app not found. Please check the file structure.")
        sys.exit(1)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend_process = start_frontend()
        
        print()
        print("✅ Both services started successfully!")
        print()
        print("🌐 Access the application at:")
        print("   Frontend: http://localhost:8501")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print()
        print("📋 Features available:")
        print("   • Upload and process documents")
        print("   • Review and edit extracted entities")
        print("   • Edit and approve document summaries")
        print("   • Submit feedback to improve accuracy")
        print("   • Export results in multiple formats")
        print()
        print("Press Ctrl+C to stop both services...")
        print()
        
        # Wait for user to stop
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
    finally:
        # Clean up processes
        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("✅ Services stopped.")

if __name__ == "__main__":
    main()

