# run_all.py
import subprocess
import sys
import os
import time

# Adjust paths if needed
FLASK_CMD = ["flask", "--app", "app", "run", "--port", "5000"]
STREAMLIT_CMD = ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]

def main():
    try:
        # Make sure environment variables are loaded
        os.environ["FLASK_ENV"] = "development"
        
        print("🚀 Starting Flask backend...")
        flask_proc = subprocess.Popen(FLASK_CMD)

        time.sleep(2)  # Give Flask a moment to start

        print("🚀 Starting Streamlit frontend...")
        streamlit_proc = subprocess.Popen(STREAMLIT_CMD)

        print("\n✅ Both servers running:")
        print("   Flask:      http://127.0.0.1:5000")
        print("   Streamlit:  http://127.0.0.1:8501")
        print("\nPress CTRL+C to stop both.\n")

        # Wait for both to finish (until user interrupts)
        flask_proc.wait()
        streamlit_proc.wait()

    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        flask_proc.terminate()
        streamlit_proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
