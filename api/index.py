import os
import sys
from streamlit.web import cli as stcli
if __name__ == '__main__':
sys.argv = ["streamlit", "run", "--server.port", os.environ.get('PORT', '8080'), "--server.headless", "true", "--server.enableCORS", "false", "your_script_name.py"]  # Replace with your Streamlit file, e.g., app.py
