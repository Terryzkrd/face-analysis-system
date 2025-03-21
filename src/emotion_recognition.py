import cv2
import mediapipe as mp
import dlib
import imutils
import numpy as np
import torchlm
import os
import pickle
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from datetime import datetime
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

class EmotionRecognition:
    def __init__(self):
        # 初始化Dlib人脸检测器和特征点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        
        # MediaPipe面网格用于更精细的特征点提取
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 历史数据队列
        self.head_pose_history = deque(maxlen=10)  # 减少历史数据量以提高性能
        self.micro_expression_history = []
        self.au_history = deque(maxlen=60)  # 存储2秒的AU数据(假设30fps)
        
        # 3D人脸模型关键点 - 用于头部姿态估计
        self.model_points_68 = self._get_full_model_points()
        
        # 相机内参（估计值）
        self.camera_matrix = np.array(
            [[840, 0, 320],
             [0, 840, 240],
             [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((4, 1))
        
        # 状态变量
        self.last_time = time.time()
        self.last_landmarks = None
        self.current_image = None
        self.emotion_confidence = {
            "Focused": 0,
            "Distracted": 0,
            "Confused": 0,
            "Fatigued": 0,
            "Excited": 0
        }
        
        # 眨眼和PERCLOS相关参数
        self.blink_counter = 0
        self.blink_start_time = time.time()
        self.eye_closed_time = 0
        self.total_time = 0
        self.perclos = 0
        self.EAR_THRESHOLD = 0.2
        self.EYE_CLOSED = False
        self.EYE_CLOSED_FRAMES = 0
        
        # 视线和瞳孔相关参数
        self.pupil_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=30)
        
        # 关注区域定义 - 可根据实际应用场景调整
        self.attention_zones = {
            "teacher": (0.4, 0.6, 0, 0.3),  # x_min, x_max, y_min, y_max (相对屏幕比例)
            "whiteboard": (0.2, 0.8, 0, 0.5)
        }
        
        # 计算眼睛3D坐标
        self._compute_eye_landmarks_3d()
        
        print("表情识别模块已初始化 v1.0")
    
    def _compute_eye_landmarks_3d(self):
        """计算眼睛在3D模型中的关键坐标，用于视线跟踪"""
        # 确保模型已经加载
        if hasattr(self, 'model_points_68'):
            # 计算眼睛中心和角点位置
            self.eye_landmarks_3d = {
                'left_center': np.mean(self.model_points_68[36:42], axis=0),
                'right_center': np.mean(self.model_points_68[42:48], axis=0),
                'left_corner': self.model_points_68[36],
                'right_corner': self.model_points_68[45]
            }
    
    def _get_full_model_points(self):
        """获取68点人脸模型的3D关键点"""
        # 使用精确的68点3D人脸模型，基于BFM和LSFM标准化人脸模型
        model_points = np.array([
            # 下巴轮廓 (0-16)
            (-73.393523, 104.55, 27.604),     # 0
            (-72.775014, 127.76, 25.110),     # 1
            (-70.533638, 150.05, 21.845),     # 2
            (-66.850058, 172.05, 19.067),     # 3
            (-59.790187, 193.02, 17.767),     # 4
            (-48.368973, 213.91, 18.458),     # 5
            (-34.121101, 227.88, 21.150),     # 6
            (-17.875411, 234.76, 24.225),     # 7
            (0.098749, 237.52, 25.610),       # 8 - 下巴尖
            (17.477031, 234.37, 24.239),      # 9
            (32.648966, 227.30, 21.314),      # 10
            (46.372358, 213.75, 18.848),      # 11
            (57.343480, 193.11, 18.210),      # 12
            (64.388482, 172.40, 19.611),      # 13
            (68.212038, 150.48, 22.438),      # 14
            (70.486405, 128.33, 25.644),      # 15
            (71.375822, 105.30, 28.044),      # 16
            
            # 眉毛 (17-26)
            (-61.119406, 85.97, 10.465),      # 17 - 左眉毛外侧
            (-48.020262, 78.37, 0.575),       # 18
            (-34.909428, 74.18, -4.825),      # 19
            (-22.003532, 72.10, -5.992),      # 20
            (-9.084950, 71.87, -5.537),       # 21 - 左眉毛内侧
            (9.314164, 71.76, -5.553),        # 22 - 右眉毛内侧
            (22.000624, 72.00, -6.096),       # 23
            (35.081570, 73.88, -5.042),       # 24
            (48.031631, 78.10, 0.134),        # 25
            (61.106239, 85.98, 9.828),        # 26 - 右眉毛外侧
            
            # 鼻子 (27-35)
            (0.000000, 93.50, 0.000),         # 27 - 鼻梁最上端
            (0.000000, 104.45, -8.860),       # 28
            (0.000000, 115.40, -15.170),      # 29
            (0.000000, 126.35, -17.600),      # 30 - 鼻尖
            (-14.561610, 126.35, -9.373),     # 31
            (-7.428795, 126.35, -12.533),     # 32
            (0.000000, 126.35, -13.330),      # 33 - 鼻子底部中心
            (7.428795, 126.35, -12.533),      # 34
            (14.561610, 126.35, -9.373),      # 35
            
            # 眼睛 (36-47)
            (-28.916267, 87.03, -12.160),     # 36 - 左眼左角
            (-17.533194, 86.87, -15.260),     # 37
            (-6.684590, 88.34, -16.276),      # 38
            (0.381003, 90.65, -15.950),       # 39 - 左眼右角
            (-6.893383, 91.05, -16.411),      # 40
            (-17.193778, 89.05, -15.890),     # 41
            (28.916267, 87.03, -12.160),      # 42 - 右眼左角
            (17.533194, 86.87, -15.260),      # 43
            (6.684590, 88.34, -16.276),       # 44
            (-0.381003, 90.65, -15.950),      # 45 - 右眼右角
            (6.893383, 91.05, -16.411),       # 46
            (17.193778, 89.05, -15.890),      # 47
            
            # 嘴巴外轮廓 (48-59)
            (-24.960880, 142.94, 7.242),      # 48 - 左嘴角
            (-15.493115, 146.54, 5.276),      # 49
            (-4.645804, 148.74, 4.732),       # 50
            (0.000000, 149.48, 4.551),        # 51
            (4.645804, 148.74, 4.732),        # 52
            (15.493115, 146.54, 5.276),       # 53
            (24.960880, 142.94, 7.242),       # 54 - 右嘴角
            (18.681839, 147.19, 8.042),       # 55
            (8.922373, 150.35, 7.268),        # 56
            (0.000000, 151.36, 7.057),        # 57 - 嘴唇下中心
            (-8.922373, 150.35, 7.268),       # 58
            (-18.681839, 147.19, 8.042),      # 59
            
            # 嘴巴内轮廓 (60-67)
            (-15.444062, 139.44, 6.041),      # 60 - 嘴左上角
            (-5.664619, 141.34, 5.554),       # 61
            (0.000000, 141.79, 5.458),        # 62 - 嘴上唇中心
            (5.664619, 141.34, 5.554),        # 63
            (15.444062, 139.44, 6.041),       # 64 - 嘴右上角
            (8.643432, 145.32, 5.811),        # 65
            (0.000000, 146.30, 5.651),        # 66 - 嘴下唇中心 
            (-8.643432, 145.32, 5.811)        # 67
        ])
        
        # 标准化处理 - 将坐标系原点调整到鼻尖(30号点)
        origin_point = model_points[30].copy()
        model_points = model_points - origin_point
        
        # 适当缩放模型以匹配一般人脸比例
        scale_factor = 0.8
        model_points *= scale_factor
        
        return model_points
    
    def get_landmarks_dlib(self, image):
        """使用dlib获取68个面部关键点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        # 获取第一个检测到的人脸的关键点
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        return landmarks, faces[0]
    
    def estimate_head_pose(self, landmarks, image):
        """使用完整68点特征点估计头部姿态"""
        if landmarks is None or len(landmarks) < 68:
            return None, None, None, None, None
        
        # 使用所有68个特征点进行PnP解算
        image_points = landmarks.astype(np.float64)
        
        # 求解PnP问题
        try:
            # 首先使用EPNP方法快速初始求解
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP)
            
            if success:
                # 使用迭代优化方法进一步精确调整
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    self.model_points_68, image_points, self.camera_matrix, self.dist_coeffs,
                    useExtrinsicGuess=True, rvec=rotation_vec, tvec=translation_vec,
                    flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception as e:
            print(f"头部姿态估计出错: {str(e)}")
            return None, None, None, None, None
        
        if not success:
            return None, None, None, None, None
        
        # 转换旋转向量为欧拉角
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        
        # 保存头部姿态历史
        self.head_pose_history.append((pitch, yaw, roll))
        
        return pitch, yaw, roll, rotation_vec, translation_vec
    
    def detect_action_units(self, landmarks, rotation_vec=None, translation_vec=None):
        """检测面部动作单元(AUs)，考虑头部姿态的影响"""
        if landmarks is None or len(landmarks) < 68:
            return {}
        
        # 计算各部位的几何特征
        aus = {}
        
        # 如果有头部姿态信息，可以进行姿态补偿
        head_rotation_compensated = False
        if rotation_vec is not None and translation_vec is not None:
            try:
                # 将关键点投影到正面视图
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                landmarks_3d = []
                
                # 对每个2D关键点，估计其3D位置
                for i, point_2d in enumerate(landmarks):
                    # 使用对应的3D模型点作为初始猜测
                    point_3d = self.model_points_68[i]
                    
                    # 将3D点从模型空间转换到相机空间
                    point_3d_camera = np.dot(rotation_mat, point_3d) + translation_vec.reshape(3)
                    
                    # 保存3D点
                    landmarks_3d.append(point_3d_camera)
                
                landmarks_3d = np.array(landmarks_3d)
                head_rotation_compensated = True
            except Exception as e:
                print(f"姿态补偿失败: {str(e)}")
        
        # 使用补偿后的关键点或原始关键点
        points = landmarks_3d if head_rotation_compensated else landmarks
        
        # AU4: 皱眉 - 计算眉间距离变化
        inner_brow_distance = np.linalg.norm(points[21] - points[22])
        brow_height = (points[21][1] + points[22][1]) / 2 - points[27][1]
        brow_height_normalized = brow_height / inner_brow_distance  # 归一化处理减少个体差异
        aus["AU4"] = brow_height_normalized < 0.4  # 皱眉时眉毛会下降
        
        # 检测左右皱眉的不对称性
        left_brow_height = points[21][1] - points[27][1]
        right_brow_height = points[22][1] - points[27][1]
        brow_asymmetry = abs(left_brow_height - right_brow_height) / inner_brow_distance
        aus["AU4_asymmetry"] = brow_asymmetry > 0.15
        
        # AU5: 上眼睑提升
        left_eye_height = np.linalg.norm(points[37] - points[41])
        right_eye_height = np.linalg.norm(points[44] - points[46])
        left_eye_width = np.linalg.norm(points[36] - points[39])
        right_eye_width = np.linalg.norm(points[42] - points[45])
        
        # 眼睛高宽比的归一化
        left_eye_ratio = left_eye_height / left_eye_width
        right_eye_ratio = right_eye_height / right_eye_width
        eye_ratio_avg = (left_eye_ratio + right_eye_ratio) / 2
        
        aus["AU5"] = eye_ratio_avg > 0.4  # 上眼睑提升时眼睛高度增加
        
        # AU6+AU7: 眯眼（Cheek Raiser + Lid Tightener）
        # 增加检测眼角皱纹的特征
        aus["AU6_7"] = eye_ratio_avg < 0.3  # 眯眼时高宽比降低
        
        # AU10: 上唇提升
        philtrum_length = np.linalg.norm(points[33] - points[51])
        upper_lip_height = np.linalg.norm(points[62] - points[51])
        aus["AU10"] = upper_lip_height / philtrum_length < 0.35
        
        # AU12: 嘴角上扬（微笑）
        mouth_corner_height = (points[54][1] + points[48][1]) / 2
        mouth_center_height = points[57][1]
        mouth_corner_dist = mouth_center_height - mouth_corner_height
        # 归一化为相对于脸部高度的比例
        face_height = np.linalg.norm(points[8] - points[27])
        aus["AU12"] = mouth_corner_dist / face_height > 0.05  # 微笑时嘴角上扬
        
        # AU15: 嘴角下拉
        aus["AU15"] = mouth_corner_dist / face_height < -0.008  # 嘴角下拉
        
        # AU25: 嘴唇分离（轻微张口）
        mouth_open = np.linalg.norm(points[62] - points[66])
        mouth_width = np.linalg.norm(points[48] - points[54])
        mouth_open_ratio = mouth_open / mouth_width
        aus["AU25"] = 0.1 < mouth_open_ratio < 0.28  # 轻微张口
        
        # AU26+AU27: 下颌下降（打哈欠）
        jaw_drop_ratio = mouth_open / face_height
        aus["AU26_27"] = jaw_drop_ratio > 0.1  # 打哈欠时下颌明显下降
        
        # 保存AU历史用于微表情检测
        self.au_history.append(aus)
        
        return aus

    def calculate_ear(self, eye_points):
        """计算眼睛的长宽比，用于眨眼检测"""
        # 计算垂直距离
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # 计算水平距离
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        # 计算EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, ear):
        """检测眨眼并更新PERCLOS指标"""
        if ear < self.EAR_THRESHOLD:
            if not self.EYE_CLOSED:
                self.EYE_CLOSED = True
            self.EYE_CLOSED_FRAMES += 1
        else:
            if self.EYE_CLOSED:
                # 成功检测到一次眨眼
                if self.EYE_CLOSED_FRAMES >= 2:  # 至少持续2帧才算眨眼
                    self.blink_counter += 1
                self.EYE_CLOSED = False
            self.EYE_CLOSED_FRAMES = 0
            
        # 更新PERCLOS - 计算60秒内眼睛闭合时间比例
        current_time = time.time()
        frame_duration = current_time - self.last_time
        self.last_time = current_time
        
        if self.EYE_CLOSED:
            self.eye_closed_time += frame_duration
            
        self.total_time += frame_duration
        
        # 每60秒重置一次计时器
        if self.total_time >= 60:
            self.perclos = self.eye_closed_time / self.total_time
            self.eye_closed_time = 0
            self.total_time = 0
            self.blink_start_time = current_time
            self.blink_counter = 0
            
        # 计算当前的眨眼频率（次/分钟）
        blink_duration = current_time - self.blink_start_time
        blink_rate = (self.blink_counter / blink_duration * 60) if blink_duration > 0 else 0
        
        return self.EYE_CLOSED, blink_rate, self.perclos
    
    def detect_iris(self, image, landmarks):
        """检测虹膜位置和直径"""
        if landmarks is None or image is None:
            return None, None
            
        # 提取左右眼区域
        def get_eye_region(eye_landmarks):
            eye_region = np.array(eye_landmarks, dtype=np.int32)
            min_x = np.min(eye_region[:, 0]) - 5
            max_x = np.max(eye_region[:, 0]) + 5
            min_y = np.min(eye_region[:, 1]) - 5
            max_y = np.max(eye_region[:, 1]) + 5
            
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(image.shape[1], max_x)
            max_y = min(image.shape[0], max_y)
            
            return (min_x, min_y, max_x, max_y)
            
        left_eye_region = get_eye_region(landmarks[36:42])
        right_eye_region = get_eye_region(landmarks[42:48])
        
        # 提取眼睛ROI
        left_eye_roi = image[left_eye_region[1]:left_eye_region[3], left_eye_region[0]:left_eye_region[2]]
        right_eye_roi = image[right_eye_region[1]:right_eye_region[3], right_eye_region[0]:right_eye_region[2]]
        
        pupils = []
        
        # 处理每只眼睛
        for eye_roi, region in [(left_eye_roi, left_eye_region), (right_eye_roi, right_eye_region)]:
            if eye_roi.size == 0 or eye_roi.shape[0] < 3 or eye_roi.shape[1] < 3:
                continue
                
            # 转换到灰度
            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            
            # 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray_eye = clahe.apply(gray_eye)
            
            # 应用高斯模糊和自适应阈值
            blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            _, thresholded = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓（可能是瞳孔）
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # 只处理合理大小的轮廓
                if area > 10:
                    # 计算直径
                    diameter = 2 * np.sqrt(area / np.pi)
                    
                    # 找到轮廓中心
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + region[0]
                        cy = int(M["m01"] / M["m00"]) + region[1]
                        pupils.append((cx, cy, diameter))
        
        # 计算平均瞳孔直径
        if pupils:
            avg_diameter = np.mean([p[2] for p in pupils])
            self.pupil_history.append(avg_diameter)
            return pupils, avg_diameter
            
        return None, None
    
    def calculate_pupil_dilation(self):
        """计算瞳孔扩张率"""
        if len(self.pupil_history) < 10:
            return 0
            
        # 获取基准瞳孔大小（使用历史数据中位数）
        baseline = np.median(list(self.pupil_history)[:10])
        current = self.pupil_history[-1]
        
        # 计算扩张率
        if baseline > 0:
            dilation_rate = (current - baseline) / baseline * 100
            return dilation_rate
        return 0
    
    def detect_gaze_direction_3d(self, landmarks, rotation_vec, translation_vec):
        """计算3D视线方向向量"""
        if landmarks is None or rotation_vec is None or translation_vec is None:
            return None, None
            
        # 检测瞳孔位置
        pupils, _ = self.detect_iris(self.current_image, landmarks)
        
        # 获取眼球3D坐标在当前头部姿态下的投影
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # 计算左右眼球中心在当前姿态下的3D位置
        left_eye_center_3d = np.dot(rotation_mat, self.eye_landmarks_3d['left_center']) + translation_vec.reshape(3)
        right_eye_center_3d = np.dot(rotation_mat, self.eye_landmarks_3d['right_center']) + translation_vec.reshape(3)
        
        # 投影到2D图像平面
        left_eye_center_2d = cv2.projectPoints(
            self.eye_landmarks_3d['left_center'].reshape(1, 3), 
            rotation_vec, translation_vec, 
            self.camera_matrix, self.dist_coeffs)[0].reshape(2)
            
        right_eye_center_2d = cv2.projectPoints(
            self.eye_landmarks_3d['right_center'].reshape(1, 3), 
            rotation_vec, translation_vec, 
            self.camera_matrix, self.dist_coeffs)[0].reshape(2)
        
        # 计算眼球中心在2D图像上的平均位置
        eye_center_2d = np.mean([left_eye_center_2d, right_eye_center_2d], axis=0)
        
        # 如果检测到瞳孔，用瞳孔中心计算视线向量
        if pupils:
            pupil_center = np.mean([(p[0], p[1]) for p in pupils], axis=0)
            
            # 从眼球中心到瞳孔中心的2D向量
            gaze_vector_2d = pupil_center - eye_center_2d
            
            # 归一化2D视线向量
            gaze_2d_norm = np.linalg.norm(gaze_vector_2d)
            if gaze_2d_norm > 0:
                gaze_vector_2d = gaze_vector_2d / gaze_2d_norm
                
            # 构造与当前头部朝向对齐的视线3D向量
            # 默认视线方向是z轴负方向（向前看）
            gaze_vector_3d = np.array([gaze_vector_2d[0], gaze_vector_2d[1], -1.0])
            gaze_vector_3d = gaze_vector_3d / np.linalg.norm(gaze_vector_3d)
            
            # 将视线向量从相机坐标系转换到世界坐标系
            gaze_vector_3d_world = np.dot(rotation_mat, gaze_vector_3d)
            
            # 存储视线历史
            self.gaze_history.append(gaze_vector_3d_world)
            
            return gaze_vector_3d_world, pupil_center
        
        # 如果未检测到瞳孔，返回头部朝向作为默认视线方向
        # 头部朝向通常是旋转矩阵的第三列（z轴方向）
        default_gaze = rotation_mat[:, 2]
        self.gaze_history.append(default_gaze)
        
        return default_gaze, eye_center_2d
    
    def is_gaze_in_zone(self, gaze_vector, zone):
        """检查视线是否在特定区域内"""
        if gaze_vector is None:
            return False
            
        # 将视线向量转换为屏幕坐标系中的方向
        # 简化实现：使用视线向量的x和y分量近似
        x_ratio = (gaze_vector[0] + 1) / 2  # 转换到0-1范围
        y_ratio = (gaze_vector[1] + 1) / 2
        
        # 检查是否在区域内
        x_min, x_max, y_min, y_max = zone
        return x_min <= x_ratio <= x_max and y_min <= y_ratio <= y_max
    
    def detect_micro_expressions(self, aus):
        """检测微表情"""
        if not aus or len(self.au_history) < 10:
            return False
            
        # 检查是否有短暂的AU激活
        # 微表情通常持续0.04-0.2秒（约1-6帧@30fps）
        micro_expr_detected = False
    
        # 将deque转换为列表再使用切片操作
        recent_history = list(self.au_history)[-10:]
        
        # 检查几个关键AU的短暂激活
        for au_name in ["AU4", "AU12", "AU6_7", "AU25"]:
            # 查找连续帧中AU的激活模式
            activations = [frame.get(au_name, False) for frame in recent_history]
            
            # 微表情特征：短暂激活后快速消失
            if (len(activations) > 0 and not activations[0] and 
                any(activations[1:5] if len(activations) > 5 else activations[1:]) and 
                not activations[-1] and sum(activations) >= 1 and sum(activations) <= 5):
                micro_expr_detected = True
                break
                
        return micro_expr_detected
    
    def detect_gaze_direction(self, landmarks):
        """基础视线方向检测（兼容原有代码）"""
        if landmarks is None or len(landmarks) < 68:
            return None
        
        # 眼球中心
        left_eye_center = np.mean([landmarks[36], landmarks[37], landmarks[38], 
                                   landmarks[39], landmarks[40], landmarks[41]], axis=0)
        right_eye_center = np.mean([landmarks[42], landmarks[43], landmarks[44], 
                                    landmarks[45], landmarks[46], landmarks[47]], axis=0)
        
        # 头部朝向（简化估计）
        face_center = landmarks[30]  # 鼻尖
        
        # 视线向量（从眼球中心到面部中心的反方向）
        gaze_vector = np.mean([left_eye_center, right_eye_center], axis=0) - face_center
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        return gaze_vector
    
    def detect_expressions(self, image):
        """检测面部表情状态，返回适合Tkinter显示的结果"""
        # 创建图像副本以避免修改原始图像
        result_image = image.copy()
        self.current_image = image.copy()  # 保存当前帧用于瞳孔检测
        
        # 获取面部关键点
        landmarks_and_face = self.get_landmarks_dlib(image)
        if landmarks_and_face is None:
            # 没有检测到人脸时，返回原始图像和提示信息
            cv2.putText(result_image, "No face detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return result_image
        
        landmarks, face_rect = landmarks_and_face
        
        # 估计头部姿态
        pitch, yaw, roll, rotation_vec, translation_vec = self.estimate_head_pose(landmarks, image)
        
        # 检测动作单元 - 使用改进的函数考虑头部姿态
        aus = self.detect_action_units(landmarks, rotation_vec, translation_vec)
        
        # 检测3D视线方向
        gaze_vector_3d, pupil_center = self.detect_gaze_direction_3d(landmarks, rotation_vec, translation_vec)
        
        # 计算眼睛开合度 (EAR - Eye Aspect Ratio)
        left_eye_ear = self.calculate_ear([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]])
        right_eye_ear = self.calculate_ear([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]])
        avg_ear = (left_eye_ear + right_eye_ear) / 2
        
        # 检测眨眼和计算PERCLOS
        is_blinking, blink_rate, perclos = self.detect_blink(avg_ear)
        
        # 计算瞳孔扩张率
        pupil_dilation_rate = self.calculate_pupil_dilation()
        
        # 检测视线是否在关注区域内
        gaze_in_teacher_zone = self.is_gaze_in_zone(gaze_vector_3d, self.attention_zones["teacher"])
        gaze_in_whiteboard_zone = self.is_gaze_in_zone(gaze_vector_3d, self.attention_zones["whiteboard"])
        
        # 检测微表情
        micro_expression = self.detect_micro_expressions(aus)
        
        # 计算动作变化（如果有之前的关键点）
        movement_magnitude = 0
        if self.last_landmarks is not None:
            # 计算面部关键点变化
            movement = np.linalg.norm(landmarks - self.last_landmarks, axis=1)
            movement_magnitude = np.mean(movement)
            
            # 微表情判断补充
            if not micro_expression:
                # 检查局部运动
                mouth_movement = np.mean(movement[48:68])  # 嘴部区域移动
                eye_movement = np.mean(movement[36:48])    # 眼睛区域移动
                brow_movement = np.mean(movement[17:27])   # 眉毛区域移动
                
                # 微表情是局部的小幅度变化
                if (1 < mouth_movement < 3 or 
                    0.5 < eye_movement < 2 or 
                    0.5 < brow_movement < 2):
                    micro_expression = True
        
        self.last_landmarks = landmarks.copy()  # 保存当前帧的关键点
        
        # 检测头部稳定性
        head_stable = True
        if len(self.head_pose_history) > 5:
            recent_yaws = [pose[1] for pose in list(self.head_pose_history)[-5:]]
            recent_pitches = [pose[0] for pose in list(self.head_pose_history)[-5:]]
            yaw_variation = np.std(recent_yaws)
            pitch_variation = np.std(recent_pitches)
            head_stable = yaw_variation < 5 and pitch_variation < 5
        
        # 重置信心值，使用指数衰减
        for emotion in self.emotion_confidence:
            self.emotion_confidence[emotion] *= 0.65
        
        # ===== 优化表情状态判断条件 =====
        
        # 1. 专注状态（Focused）
        # 凝视方向稳定+眨眼频率较低+轻微皱眉
        focused_conditions = [
            (yaw is not None and abs(yaw) < 20),           # 头部角度小
            (gaze_in_teacher_zone or gaze_in_whiteboard_zone), # 视线在关注区域
            head_stable,                                   # 头部稳定
            (blink_rate < 15),                             # 眨眼频率低于15次/分钟
            aus.get("AU4", False),                         # 轻微皱眉（专注表现）
            not micro_expression                           # 没有微表情干扰
        ]
        if sum(focused_conditions) >= 4:  # 满足大部分条件
            self.emotion_confidence["Focused"] += 0.35
        
        # 2. 分心状态（Distraction）
        # 头部偏转角度大+视线频繁扫视非教学区域+微表情
        distracted_conditions = [
            (yaw is not None and abs(yaw) > 20),           # 头部转向
            not (gaze_in_teacher_zone or gaze_in_whiteboard_zone), # 视线不在关注区域
            not head_stable,                               # 头部不稳定
            micro_expression,                              # 存在微表情
            movement_magnitude > 2                         # 频繁移动
        ]
        if sum(distracted_conditions) >= 3:
            self.emotion_confidence["Distracted"] += 0.35
        
        # 3. 困惑状态（Confusion）
        # 单侧皱眉+频繁眯眼+嘴唇轻微张开
        confused_conditions = [
            aus.get("AU4_asymmetry", False),               # 单侧皱眉（表示思考/困惑）
            aus.get("AU6_7", False),                       # 眯眼
            aus.get("AU25", False),                        # 嘴唇轻微张开
            not (gaze_in_teacher_zone and gaze_in_whiteboard_zone) # 视线不完全专注
        ]
        if sum(confused_conditions) >= 2:
            self.emotion_confidence["Confused"] += 0.35
        
        # 4. 疲劳状态（Fatigue）
        # PERCLOS过高+头部下倾+打哈欠
        fatigued_conditions = [
            (perclos > 0.3),                               # PERCLOS指标高
            (pitch is not None and pitch < -10),           # 头部下倾
            avg_ear < 0.2,                                 # 眼睛半闭
            aus.get("AU26_27", False)                      # 打哈欠
        ]
        if sum(fatigued_conditions) >= 2:
            self.emotion_confidence["Fatigued"] += 0.35
        
        # 5. 兴奋状态（Excitement）
        # 嘴角上扬+上眼睑提升+瞳孔扩大
        excited_conditions = [
            aus.get("AU12", False),                        # 嘴角上扬（微笑）
            aus.get("AU5", False),                         # 上眼睑提升
            (pupil_dilation_rate > 8),                     # 瞳孔扩大
            avg_ear > 0.30,                                # 眼睛睁大
            movement_magnitude > 1                         # 面部活跃
        ]
        if sum(excited_conditions) >= 3:
            self.emotion_confidence["Excited"] += 0.35
            
        # 确定最高可能的情绪状态
        max_emotion = max(self.emotion_confidence, key=self.emotion_confidence.get)
        max_confidence = self.emotion_confidence[max_emotion]
        
        # 如果信心值太低，显示为未知
        if max_confidence < 0.5:
            max_emotion = "Unknown"
        
        # === 绘制可视化信息 ===
        
        # 绘制面部关键点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
        
        # 绘制面部边界框
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 显示头部姿态
        if rotation_vec is not None and translation_vec is not None:
            # 计算头部前方的点以显示朝向
            nose_end_point2D = cv2.projectPoints(
                np.array([(0, 0, 200.0)]), rotation_vec, translation_vec, 
                self.camera_matrix, self.dist_coeffs)[0]
            
            p1 = (int(landmarks[30][0]), int(landmarks[30][1]))  # 鼻尖
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            cv2.line(result_image, p1, p2, (0, 0, 255), 2)
            
        # 显示视线方向
        if gaze_vector_3d is not None and pupil_center is not None:
            # 视线向量可视化
            gaze_length = 100
            gaze_point = (
                int(pupil_center[0] + gaze_vector_3d[0] * gaze_length), 
                int(pupil_center[1] + gaze_vector_3d[1] * gaze_length)
            )
            cv2.line(result_image, (int(pupil_center[0]), int(pupil_center[1])), 
                    gaze_point, (255, 0, 255), 2)
        
        # 显示瞳孔中心（如果检测到）
        pupils, _ = self.detect_iris(image, landmarks)
        if pupils:
            for (x, y, d) in pupils:
                cv2.circle(result_image, (int(x), int(y)), int(d/2), (0, 255, 255), 1)
                cv2.circle(result_image, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        # 显示表情状态
        cv2.putText(result_image, f"Emotion: {max_emotion}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 显示信心度
        info_y = 60
        info_spacing = 25
        
        cv2.putText(result_image, f"Confidence: {max_confidence:.2f}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        info_y += info_spacing
        
        # 显示眨眼和PERCLOS信息
        cv2.putText(result_image, f"Blink Rate: {blink_rate:.1f}/min", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        info_y += info_spacing
        
        cv2.putText(result_image, f"PERCLOS: {perclos:.2f}", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        info_y += info_spacing
        
        # 显示头部姿态信息
        if yaw is not None:
            cv2.putText(result_image, f"Head: P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}", 
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            info_y += info_spacing
        
        # 显示检测到的AU特征
        active_aus = [au for au, active in aus.items() if active]
        if active_aus:
            # 将过长的AU列表分割显示
            if len(active_aus) > 5:
                first_line = active_aus[:5]
                second_line = active_aus[5:]
                cv2.putText(result_image, f"AUs: {', '.join(first_line)}", 
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                info_y += info_spacing
                cv2.putText(result_image, f"     {', '.join(second_line)}", 
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(result_image, f"AUs: {', '.join(active_aus)}", 
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return result_image

