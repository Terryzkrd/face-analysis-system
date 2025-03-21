import unittest
from src.emotion_recognition import EmotionRecognition

class TestEmotionRecognition(unittest.TestCase):
    def setUp(self):
        self.emotion_recognition = EmotionRecognition()

    def test_initialization(self):
        self.assertIsNotNone(self.emotion_recognition)

    def test_get_landmarks_dlib_no_face(self):
        image = ...  # Load or create a test image with no face
        landmarks = self.emotion_recognition.get_landmarks_dlib(image)
        self.assertIsNone(landmarks)

    def test_detect_expressions(self):
        image = ...  # Load or create a test image with a face
        result_image, emotion_state = self.emotion_recognition.detect_expressions(image)
        self.assertIn(emotion_state, ["Focused", "Distracted", "Confused", "Fatigued", "Excited", "Unknown"])

    def test_detect_action_units(self):
        image = ...  # Load or create a test image with a face
        landmarks, _ = self.emotion_recognition.get_landmarks_dlib(image)
        action_units = self.emotion_recognition.detect_action_units(landmarks)
        self.assertIsInstance(action_units, dict)

    def test_calculate_ear(self):
        eye_points = ...  # Define test eye points
        ear = self.emotion_recognition.calculate_ear(eye_points)
        self.assertIsInstance(ear, float)

if __name__ == '__main__':
    unittest.main()