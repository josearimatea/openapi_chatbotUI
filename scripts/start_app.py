# scripts/start_app.py
"""
Utility script to start both backend (FastAPI) and frontend (Streamlit) in parallel.
Uses subprocess to run commands concurrently, with support for graceful shutdown.
Run with: uv run python scripts/start_app.py
"""

import subprocess
import signal
import sys
import time

# Define the commands to run (as lists for subprocess)
BACKEND_CMD = [
    "uv", "run", "uvicorn", "app.main:app",
    "--reload", "--port", "5000"
]

FRONTEND_CMD = [
    "uv", "run", "streamlit", "run", "frontend/app.py",
    "--server.port", "5001"
]

def start_process(cmd):
    """Start a subprocess with the given command and return the process."""
    return subprocess.Popen(cmd)

def main():
    """Main function to start both services and handle shutdown."""
    print("Starting backend and frontend...")
    
    # Start backend
    backend_proc = start_process(BACKEND_CMD)
    print(f"Backend started (PID: {backend_proc.pid})")
    
    # Give a short delay to avoid port conflicts or output overlap
    time.sleep(2)
    
    # Start frontend
    frontend_proc = start_process(FRONTEND_CMD)
    print(f"Frontend started (PID: {frontend_proc.pid})")
    
    # Handle shutdown (CTRL+C)
    def shutdown(signal, frame):
        print("\nShutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("Shutdown complete.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    
    # Wait for processes to finish (they run indefinitely until killed)
    backend_proc.wait()
    frontend_proc.wait()

if __name__ == "__main__":
    main()