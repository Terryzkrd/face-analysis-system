# Usage Instructions for Face Analysis System

## Overview
The Face Analysis System is designed to perform real-time face recognition and emotion analysis using webcam input. This document provides instructions on how to use the system effectively.

## Prerequisites
Before using the system, ensure that you have completed the setup as described in `setup.md`. You will need:
- A webcam connected to your computer.
- The required Python packages installed as listed in `requirements.txt`.

## Running the Application
To start the application, run the following command in your terminal:

```
python main.py
```

This will launch the user interface where you can interact with the face recognition and emotion analysis features.

## Features

### Face Recognition
1. **Register a New Face**:
   - Navigate to the "Face Recognition" tab.
   - Upload an image of the face you want to register.
   - Enter a name or ID for the face.
   - Click the "Register Face" button to save the face in the database.

2. **Recognize a Face**:
   - Upload an image containing a face.
   - Click the "Recognize Face" button to identify the person in the image.

3. **Manage Registered Faces**:
   - Click the "List Registered Faces" button to view all registered faces.
   - To delete a face, enter the name of the face in the provided textbox and click "Delete Face".

### Emotion Detection
1. **Real-time Emotion Analysis**:
   - Navigate to the "Emotion Detection" tab.
   - Allow access to your webcam.
   - The system will automatically analyze your facial expressions and display the detected emotion in real-time.

## Supported Emotions
The system can detect the following emotions:
- Focused
- Distracted
- Confused
- Fatigued
- Excited

## Example Usage
1. **Registering a Face**:
   - Upload an image of John Doe.
   - Enter "John Doe" in the name field.
   - Click "Register Face".

2. **Recognizing a Face**:
   - Upload an image of John Doe.
   - Click "Recognize Face".
   - The system should display "ID: John Doe".

3. **Emotion Detection**:
   - Allow webcam access.
   - The system will display your current emotional state based on facial expressions.

## Troubleshooting
- If the webcam does not work, ensure it is properly connected and recognized by your operating system.
- For any errors during face recognition or emotion detection, check the console for error messages and ensure all dependencies are installed correctly.

## Conclusion
The Face Analysis System provides a powerful tool for face recognition and emotion analysis. Follow the instructions above to utilize its features effectively. For further assistance, refer to the `README.md` or contact the support team.