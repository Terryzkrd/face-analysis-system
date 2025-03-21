# Setup Instructions for Face Analysis System

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.6 or higher
- pip (Python package installer)

## Installation Steps

1. **Clone the Repository**

   Open your terminal and clone the repository using the following command:

   ```
   git clone <repository-url>
   ```

   Replace `<repository-url>` with the actual URL of the repository.

2. **Navigate to the Project Directory**

   Change your directory to the project folder:

   ```
   cd face-analysis-system
   ```

3. **Install Required Packages**

   Install the necessary dependencies using pip. Run the following command:

   ```
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**

   Ensure that the following pre-trained models are available in the `models` directory:

   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`

   If these files are not present, download them from the respective sources and place them in the `models` directory.

5. **Set Up the Face Database**

   The face database is stored in `data/face_database.pkl`. If you are starting fresh, this file will be created automatically when you register a new face using the application.

## Running the Application

To run the application, execute the following command:

```
python main.py
```

This will start the face analysis system, and you can access the user interface through your web browser.

## Additional Information

For usage instructions and examples, refer to the `docs/usage.md` file.