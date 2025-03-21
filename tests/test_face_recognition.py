import unittest
from src.face_recognition import FaceRecognition
import cv2
import numpy as np

class TestFaceRecognition(unittest.TestCase):

    def setUp(self):
        self.face_recognition = FaceRecognition()
        self.test_image = cv2.imread('tests/test_image.jpg')  # Placeholder for a test image

    def test_load_faces(self):
        self.face_recognition.load_faces()
        self.assertIsInstance(self.face_recognition.face_database, dict)

    def test_register_face(self):
        result_image, message = self.face_recognition.register_face(self.test_image, "TestUser")
        self.assertIn("注册成功", message)

    def test_recognize_face(self):
        self.face_recognition.register_face(self.test_image, "TestUser")
        result_image, message = self.face_recognition.recognize_face(self.test_image)
        self.assertIn("ID: TestUser", message)

    def test_extract_face_features(self):
        face_descriptor, face_rect, message = self.face_recognition.extract_face_features(self.test_image)
        self.assertIsNotNone(face_descriptor)
        self.assertIsInstance(face_rect, tuple)

    def test_delete_face(self):
        self.face_recognition.register_face(self.test_image, "TestUser")
        delete_message = self.face_recognition.delete_face("TestUser")
        self.assertIn("已删除", delete_message)

if __name__ == '__main__':
    unittest.main()