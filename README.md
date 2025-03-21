# Face Analysis System

This project is a comprehensive face analysis system that integrates face recognition and emotion detection functionalities. It utilizes advanced machine learning models and computer vision techniques to analyze facial features and expressions in real-time.

## Project Structure

```
face-analysis-system
├── src
│   ├── core
│   │   ├── __init__.py
│   │   ├── face_recognition.py
│   │   ├── emotion_recognition.py
│   │   └── utils.py
│   ├── ui
│   │   ├── __init__.py
│   │   └── app.py
│   └── __init__.py
├── models
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
├── data
│   └── face_database.pkl
├── tests
│   ├── __init__.py
│   ├── test_face_recognition.py
│   └── test_emotion_recognition.py
├── docs
│   ├── setup.md
│   └── usage.md
├── requirements.txt
├── main.py
└── README.md
```

## Features

- **Face Recognition**: Detect and recognize faces using pre-trained models.
- **Emotion Detection**: Analyze facial expressions to determine emotional states.
- **Real-time Processing**: Utilize webcam input for live analysis.
- **User Interface**: A user-friendly interface built with Gradio for easy interaction.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-analysis-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the necessary models and place them in the `models` directory.

## Usage

To run the application, execute the following command:
```
python main.py
```

This will launch the Gradio interface where you can register faces, recognize them, and analyze emotions in real-time.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.