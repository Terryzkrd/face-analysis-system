import os
from src.app import create_ui

if __name__ == '__main__':
    # Create the necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    os.makedirs('tests', exist_ok=True)
    os.makedirs('docs', exist_ok=True)

    # Launch the user interface for face and emotion analysis
    create_ui()