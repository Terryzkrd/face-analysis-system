import cv2
import numpy as np
import time
from collections import deque
import os
import matplotlib.pyplot as plt
from datetime import datetime
from face_recognition import FaceRecognition
from emotion_recognition import EmotionRecognition

class ClassroomMonitor:
    """学生课堂状态监测系统"""
    
    def __init__(self):
        # 初始化人脸识别和表情识别模块
        self.face_recognition = FaceRecognition()
        self.emotion_recognition = EmotionRecognition()

        # 添加数据收集结构
        self.student_understanding_data = {}  # 格式: {student_id: [(timestamp, score), ...]}
        self.session_start_time = None
        self.frame_count = 0  # 帧计数器，用于控制数据采样频率
        self.sampling_rate = 1  # 每X帧采样一次数据
        
        # 理解度历史记录
        self.understanding_history = {}  # 按学生ID存储
        self.history_length = 10         # 历史记录长度
        
        # 平滑系数
        self.smoothing_alpha = 0.3
        
        # 界面设置
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (50, 205, 50)     # 绿色
        self.alert_color = (0, 0, 255)         # 红色
        self.neutral_color = (255, 255, 255)   # 白色
        self.header_color = (255, 215, 0)      # 金色
        
        # 状态标签映射
        self.emotion_labels = {
            "Focused": "Focused",
            "Distracted": "Distracted",
            "Confused": "Confused",
            "Fatigued": "Fatigued", 
            "Excited": "Excited",
            "Unknown": "Unknown"
        }
        
        print("课堂状态监测模块已初始化 v1.0")
    
    def calculate_understanding_score(self, emotion_scores):
        """
        基于五种情绪状态计算学生的课程理解度评分
        """
        # 设定各情绪状态对理解度的权重系数
        weights = {
            "Focused": 0.50,     # 专注是理解的最大正向因素
            "Distracted": 0.05, # 分心严重影响理解
            "Confused": 0.15,   # 困惑表示理解障碍，但可能是思考过程
            "Fatigued": 0.10,   # 疲劳降低认知能力
            "Excited": 0.20      # 适度兴奋有助于理解和记忆
        }
        
        # 计算加权得分
        weighted_score = 0
        for emotion, score in emotion_scores.items():
            if emotion in weights:
                weighted_score += weights[emotion] * score
        
        # 限制分数范围在0-100之间
        final_score = weighted_score
        
        return final_score
    
    def smooth_understanding_score(self, student_id, current_score):
        """平滑处理理解度分数，避免剧烈波动"""
        if student_id not in self.understanding_history:
            self.understanding_history[student_id] = deque(maxlen=self.history_length)
            
        history = self.understanding_history[student_id]
        history.append(current_score)
        
        # 如果历史记录不足，直接返回当前分数
        if len(history) < 3:
            return current_score
            
        # 应用指数加权移动平均
        alpha = self.smoothing_alpha
        smoothed_score = current_score * alpha + (1 - alpha) * sum(list(history)[:-1]) / (len(history) - 1)
        
        return smoothed_score
    
    def process_frame(self, frame):
        """处理视频帧，返回增强显示的帧"""
        # 如果是第一帧，记录开始时间
        if self.session_start_time is None:
            self.session_start_time = time.time()
        
        self.frame_count += 1
        result_image = frame.copy()
        
        # 获取面部关键点和边界框
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_recognition.detector(gray, 0)
        
        if len(faces) == 0:
            cv2.putText(result_image, "No student detect", (20, 50), 
                      self.display_font, 1.0, self.alert_color, 2)
            return result_image
        
        # 添加状态监测标题
        cv2.putText(result_image, "student_monitor_system", (20, 30),
                    self.display_font, 1.0, self.header_color, 2)
        
        # 在每个检测到的人脸上执行分析
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
            # 1. 使用优化后的人脸识别方法
            student_id, confidence, landmarks = self.face_recognition.identify_face(frame, face, gray)
            
            # 2. 表情识别 - 使用已识别的面部特征点
            landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            # 更新情绪状态
            self.emotion_recognition.detect_expressions(frame)
        
            # 获取更新后的情绪分数
            emotion_scores = self.emotion_recognition.emotion_confidence
        
            # 识别主要情绪 - 使用自定义阈值
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_confidence = emotion_scores[max_emotion]
            
            if max_confidence < 60:
                max_emotion = "Unknown"
            
            emotion_label = self.emotion_labels.get(max_emotion, "Unknown")
            
            # 3. 计算理解度
            understanding_score = self.calculate_understanding_score(emotion_scores)
            smoothed_score = self.smooth_understanding_score(student_id, understanding_score)
        
            # 收集数据 - 每X帧采样一次，避免数据过多
            if self.frame_count % self.sampling_rate == 0 and student_id != "Unknown":
                current_time = time.time() - self.session_start_time  # 相对时间(秒)
                
                if student_id not in self.student_understanding_data:
                    self.student_understanding_data[student_id] = []
                
                self.student_understanding_data[student_id].append((current_time, smoothed_score))
            
            # 绘制结果
            # 1. 面部边界框
            box_color = self.primary_color if max_emotion == "Focused" else (
                         self.alert_color if max_emotion in ["Distracted", "Fatigued"] else 
                         self.neutral_color)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, 2)
            
            # 计算信息框位置
            info_x = min(x1, frame.shape[1] - 200)
            info_y = min(y2 + 10, frame.shape[0] - 90)
            
            # 绘制信息框
            cv2.rectangle(result_image, (info_x, info_y), (info_x + 180, info_y + 85), (45, 45, 45), -1)
            cv2.rectangle(result_image, (info_x, info_y), (info_x + 180, info_y + 85), box_color, 2)
            
            # 只显示三项关键信息
            cv2.putText(result_image, f"ID: {student_id}", (info_x + 10, info_y + 20), 
                      self.display_font, 0.5, (255, 255, 255), 1)
            cv2.putText(result_image, f"Emotion: {emotion_label}", (info_x + 10, info_y + 45), 
                      self.display_font, 0.5, self.neutral_color, 1)
            cv2.putText(result_image, f"Score: {smoothed_score:.1f}", (info_x + 10, info_y + 70), 
                      self.display_font, 0.5, self.neutral_color, 1)
        
        return result_image
    
    def generate_understanding_reports(self, output_directory="reports"):
        """仅生成每个学生的理解度趋势图"""
        # 创建输出目录
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # 生成报告时间戳
        report_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_count = 0
        
        # 为每个学生生成趋势图
        for student_id, data_points in self.student_understanding_data.items():
            # 跳过数据点太少的学生
            if len(data_points) < 5:
                continue
                
            # 解析数据
            timestamps = [point[0] for point in data_points]
            scores = [point[1] for point in data_points]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, scores, 'b-', linewidth=2)
            plt.fill_between(timestamps, 0, scores, alpha=0.2)
            
            # 添加标题和标签
            plt.title(f"Student Understanding Trend - {student_id}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Understanding Score")
            plt.ylim(0, max(100, max(scores) + 10))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加统计信息
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            stats_text = f"Avg: {avg_score:.1f}\nMax: {max_score:.1f}\nMin: {min_score:.1f}"
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        verticalalignment='top')
            
            # 保存图表
            filename = f"{output_directory}/{report_datetime}_{student_id}.png"
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            
            report_count += 1
        
        return report_count
    