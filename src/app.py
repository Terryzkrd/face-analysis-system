import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import os
import sys
import threading
import time
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from emotion_recognition import EmotionRecognition
from face_recognition import FaceRecognition
from student_monitor import ClassroomMonitor

plt.rcParams['font.sans-serif'] = ['SimHei']  
# 使用SimHei字体这样可以显示title中的中文，但这个不显示负号
plt.rcParams['axes.unicode_minus'] = False  
# 解决负号显示问题

class FaceAnalysisApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # 创建识别对象
        self.face_recognition = FaceRecognition()
        self.emotion_recognition = EmotionRecognition()
        
        # 创建标签页
        self.tab_control = ttk.Notebook(window)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='人脸注册与识别')
        self.tab_control.add(self.tab2, text='实时人脸识别')
        self.tab_control.add(self.tab3, text='表情状态检测')
        self.tab_control.add(self.tab4, text='课堂状态监测')
        self.tab_control.pack(expand=1, fill="both")
        
        # 设置各个标签页的内容
        self.setup_registration_tab()
        self.setup_live_recognition_tab()
        self.setup_emotion_detection_tab()
        self.setup_classroom_monitor_tab()
        
        # 视频捕获变量
        self.cap = None
        self.is_capturing = False
        
        # 定义关闭窗口时的行为
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 启动主循环
        self.window.mainloop()
    
    def setup_registration_tab(self):
        # 左侧: 图像显示和文件选择
        left_frame = ttk.Frame(self.tab1)
        left_frame.pack(side="left", padx=10, pady=10)
        
        # 图像显示区域
        self.canvas = tk.Canvas(left_frame, width=400, height=300)
        self.canvas.pack(fill="both", expand=True)
    
        # 图像选择按钮和拍照按钮框架
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(pady=5, fill="x")
    
        # 图像选择按钮
        btn_select = ttk.Button(buttons_frame, text="选择图像", command=self.select_image)
        btn_select.pack(side="left", padx=5, expand=True)
    
        # 拍照按钮
        btn_camera = ttk.Button(buttons_frame, text="摄像头拍照", command=self.capture_photo)
        btn_camera.pack(side="left", padx=5, expand=True)
        
        # 右侧: 注册和识别功能
        right_frame = ttk.Frame(self.tab1)
        right_frame.pack(side="right", padx=10, pady=10, fill="y")
        
        # 注册区域
        register_frame = ttk.LabelFrame(right_frame, text="注册人脸")
        register_frame.pack(pady=5, fill="x")
        
        ttk.Label(register_frame, text="人脸名称/ID:").pack(pady=5)
        self.face_name_var = tk.StringVar()
        ttk.Entry(register_frame, textvariable=self.face_name_var).pack(pady=5)
        ttk.Button(register_frame, text="注册", command=self.register_face).pack(pady=5)
        
        # 识别区域
        recognize_frame = ttk.LabelFrame(right_frame, text="识别人脸")
        recognize_frame.pack(pady=5, fill="x")
        ttk.Button(recognize_frame, text="识别", command=self.recognize_face).pack(pady=5)
        self.recognition_result_var = tk.StringVar()
        ttk.Label(recognize_frame, textvariable=self.recognition_result_var).pack(pady=5)
        
        # 数据库管理区域
        db_frame = ttk.LabelFrame(right_frame, text="人脸数据库管理")
        db_frame.pack(pady=5, fill="x")
        ttk.Button(db_frame, text="列出已注册人脸", command=self.list_faces).pack(pady=5)
        self.faces_list = tk.Text(db_frame, height=10, width=30)
        self.faces_list.pack(pady=5)
        ttk.Label(db_frame, text="要删除的人脸:").pack(pady=5)
        self.delete_name_var = tk.StringVar()
        ttk.Entry(db_frame, textvariable=self.delete_name_var).pack(pady=5)
        ttk.Button(db_frame, text="删除", command=self.delete_face).pack(pady=5)

    def capture_photo(self):
        """打开摄像头并拍照"""
        # 创建一个顶层窗口用于显示摄像头预览
        camera_window = tk.Toplevel(self.window)
        camera_window.title("摄像头拍照")
        camera_window.geometry("640x520")
        
        # 创建显示摄像头预览的画布
        preview_canvas = tk.Canvas(camera_window, width=640, height=480)
        preview_canvas.pack(pady=5)
        
        # 创建拍照按钮
        photo_button = ttk.Button(camera_window, text="拍照", width=20)
        photo_button.pack(pady=5)
        
        # 开启摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            camera_window.destroy()
            return
        
        # 用于存储拍摄的照片
        captured_image = [None]
        
        def update_preview():
            ret, frame = cap.read()
            if ret:
                # 显示预览
                cv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PIL.Image.fromarray(cv_image_rgb)
                tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
                preview_canvas.image = tk_image
                preview_canvas.create_image(320, 240, image=tk_image)
                
                # 如果窗口仍然存在，继续更新
                if camera_window.winfo_exists():
                    camera_window.after(10, update_preview)
                else:
                    cap.release()
        
        def take_photo():
            """拍照并关闭窗口"""
            ret, frame = cap.read()
            if ret:
                captured_image[0] = frame.copy()
                cap.release()
                camera_window.destroy()
                
                # 更新主界面上的图像
                self.current_image = captured_image[0]
                self.display_image(self.current_image, self.canvas)
        
        # 设置拍照按钮事件
        photo_button.config(command=take_photo)
        
        # 添加窗口关闭事件
        def on_window_close():
            cap.release()
            camera_window.destroy()
        
        camera_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # 开始预览
        update_preview()
    
    def setup_live_recognition_tab(self):
        # 视频显示区域
        self.video_canvas = tk.Canvas(self.tab2, width=640, height=480)
        self.video_canvas.pack(pady=10)
        
        # 控制按钮
        control_frame = ttk.Frame(self.tab2)
        control_frame.pack(pady=5)
        
        self.start_recognition_btn = ttk.Button(control_frame, text="开始识别", command=self.start_live_recognition)
        self.start_recognition_btn.pack(side="left", padx=5)
        
        self.stop_recognition_btn = ttk.Button(control_frame, text="停止", command=self.stop_capture)
        self.stop_recognition_btn.pack(side="left", padx=5)
    
    def setup_emotion_detection_tab(self):
        # 视频显示区域
        self.emotion_canvas = tk.Canvas(self.tab3, width=640, height=480)
        self.emotion_canvas.pack(pady=10)
        
        # 控制按钮
        control_frame = ttk.Frame(self.tab3)
        control_frame.pack(pady=5)
        
        self.start_emotion_btn = ttk.Button(control_frame, text="开始检测", command=self.start_emotion_detection)
        self.start_emotion_btn.pack(side="left", padx=5)
        
        self.stop_emotion_btn = ttk.Button(control_frame, text="停止", command=self.stop_capture)
        self.stop_emotion_btn.pack(side="left", padx=5)
        
        # 表情状态说明
        info_frame = ttk.LabelFrame(self.tab3, text="检测的表情状态")
        info_frame.pack(pady=10, fill="x", padx=10)
        
        info_text = """
        1. Focused: 专注状态，凝视方向稳定，眨眼频率低，轻微皱眉
        2. Distracted: 分心状态，头部偏转角度大，视线频繁扫视，微表情变化快
        3. Confused: 困惑状态，单侧皱眉，频繁眯眼，嘴唇轻微张开
        4. Fatigued: 疲劳状态，PERCLOS指标高，头部下倾，打哈欠动作
        5. Excited: 兴奋状态，嘴角上扬，上眼睑提升，瞳孔扩大
        """
        ttk.Label(info_frame, text=info_text, justify="left").pack(pady=5)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image, self.canvas)
    
    def display_image(self, cv_image, canvas):
        # 转换OpenCV图像为Tkinter可显示的格式
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(cv_image_rgb)
        
        # 调整图像大小以适应canvas
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 300
        
        pil_image = self.resize_image(pil_image, canvas_width, canvas_height)
        
        # 创建PhotoImage对象
        tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
        
        # 保存引用以防止图像被垃圾回收
        canvas.image = tk_image
        
        # 显示图像
        canvas.create_image(canvas_width//2, canvas_height//2, image=tk_image)
    
    def resize_image(self, pil_image, width, height):
        # 计算调整比例
        w, h = pil_image.size
        aspect_ratio = min(width/w, height/h)
        new_size = (int(w * aspect_ratio), int(h * aspect_ratio))
        
        return pil_image.resize(new_size, PIL.Image.LANCZOS)
    
    def register_face(self):
        name = self.face_name_var.get()
        if hasattr(self, 'current_image') and name:
            result_image, message = self.face_recognition.register_face(self.current_image, name)
            self.display_image(result_image, self.canvas)
            messagebox.showinfo("注册结果", message)
        else:
            messagebox.showerror("错误", "请选择图像并输入名称")
    
    def recognize_face(self):
        if hasattr(self, 'current_image'):
            result_image, message = self.face_recognition.recognize_face(self.current_image)
            self.display_image(result_image, self.canvas)
            self.recognition_result_var.set(message)
        else:
            messagebox.showerror("错误", "请先选择图像")
    
    def list_faces(self):
        faces_text = self.face_recognition.list_registered_faces()
        self.faces_list.delete(1.0, tk.END)
        self.faces_list.insert(tk.END, faces_text)
    
    def delete_face(self):
        name = self.delete_name_var.get()
        if name:
            result = self.face_recognition.delete_face(name)
            messagebox.showinfo("删除结果", result)
            self.list_faces()  # 刷新列表
        else:
            messagebox.showerror("错误", "请输入要删除的人脸名称")
    
    def start_capture(self, target_canvas, process_func):
        if self.is_capturing:
            self.stop_capture()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        self.is_capturing = True
        self.target_canvas = target_canvas
        self.process_func = process_func
        
        # 在新线程中启动视频捕获
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    def update_frame(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame = self.process_func(frame)
            
            # 在主线程中更新UI
            self.window.after(1, lambda: self.display_image(processed_frame, self.target_canvas))
            
            # 控制帧率
            time.sleep(0.03)  # 约30fps
    
    def start_live_recognition(self):
        self.start_capture(self.video_canvas, self.face_recognition.recognize_face_stream)
    
    def start_emotion_detection(self):
        self.start_capture(self.emotion_canvas, self.emotion_recognition.detect_expressions)
    
    def stop_capture(self):
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
    
    def on_closing(self):
        self.stop_capture()
        self.window.destroy()

    def setup_classroom_monitor_tab(self):
        """设置课堂状态监测标签页"""
        # 创建课堂监测对象
        self.classroom_monitor = ClassroomMonitor()
        
        # 视频显示区域
        self.monitor_canvas = tk.Canvas(self.tab4, width=800, height=600)
        self.monitor_canvas.pack(pady=10)
        
        # 控制按钮框架
        control_frame = ttk.Frame(self.tab4)
        control_frame.pack(pady=5)
        
        # 开始监测按钮
        self.start_monitor_btn = ttk.Button(
            control_frame, 
            text="开始课堂监测", 
            command=lambda: self.start_capture(self.monitor_canvas, self.classroom_monitor.process_frame)
        )
        self.start_monitor_btn.pack(side="left", padx=5)
        
        # 停止按钮
        self.stop_monitor_btn = ttk.Button(
            control_frame, 
            text="停止监测", 
            command=self.stop_capture
        )
        self.stop_monitor_btn.pack(side="left", padx=5)
        
        # 生成理解度趋势图按钮
        self.generate_trend_btn = ttk.Button(
            control_frame, 
            text="生成理解度趋势图", 
            command=self.generate_understanding_trends
        )
        self.generate_trend_btn.pack(side="left", padx=5)
        
        # Matplotlib 图表
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.ax.set_title("学生理解度趋势")
        self.ax.set_xlabel("时间 (秒)")
        self.ax.set_ylabel("理解度分数 (0-100)")
        self.ax.set_ylim(0, 100)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab4)
        self.canvas.get_tk_widget().pack(pady=10)
        
        self.is_monitoring = False  # 监测状态
        # 添加说明文本
        info_frame = ttk.LabelFrame(self.tab4, text="课堂状态监测说明")
        info_frame.pack(pady=10, fill="x", padx=10)
        
        info_text = """
        课堂状态监测系统使用说明:
        
        1. 点击"开始课堂监测"收集学生理解度数据
        2. 监测过程中，可以实时查看每位学生的情绪状态和理解度分数
        3. 完成监测后，点击"生成理解度趋势图"生成每位学生的理解度变化图表
        """
        ttk.Label(info_frame, text=info_text, justify="left").pack(pady=5)

    def generate_understanding_trends(self):
        """在 Tkinter 界面中显示每个学生的理解度趋势"""
        if not self.classroom_monitor.student_understanding_data:
            messagebox.showwarning("无法生成趋势图", "没有足够的数据生成趋势图。\n请确保监测时间足够长，至少有5个数据点。")
            return

        # 创建新窗口用于显示趋势图
        trend_window = tk.Toplevel(self.window)
        trend_window.title("学生理解度趋势")
        trend_window.geometry("800x500")

        # 创建 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("学生理解度趋势")
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("理解度分数 (0-100)")
        ax.set_ylim(0, 100)

        for student_id, data_points in self.classroom_monitor.student_understanding_data.items():
            timestamps = [point[0] for point in data_points]
            scores = [point[1] for point in data_points]
            ax.plot(timestamps, scores, marker="o", linestyle="-", label=f"学生 {student_id}")

        ax.legend()

        # 嵌入 Tkinter Canvas
        canvas = FigureCanvasTkAgg(fig, master=trend_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def start_monitoring(self):
        """开始课堂监测并实时更新理解度趋势"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.classroom_monitor.student_understanding_data = {}  # 清空旧数据
        self.ax.clear()

        def update_graph():
            start_time = time.time()
            while self.is_monitoring:
                elapsed_time = time.time() - start_time
                student_id = random.choice(["S1", "S2", "S3"])  # 模拟学生 ID
                understanding_score = random.randint(50, 100)  # 模拟数据，可替换成实际分析值

                if student_id not in self.classroom_monitor.student_understanding_data:
                    self.classroom_monitor.student_understanding_data[student_id] = []
                self.classroom_monitor.student_understanding_data[student_id].append((elapsed_time, understanding_score))

                # 更新 Matplotlib 图表
                self.ax.clear()
                self.ax.set_title("学生理解度趋势")
                self.ax.set_xlabel("时间 (秒)")
                self.ax.set_ylabel("理解度分数 (0-100)")
                self.ax.set_ylim(0, 100)

                for student_id, data_points in self.classroom_monitor.student_understanding_data.items():
                    timestamps = [point[0] for point in data_points]
                    scores = [point[1] for point in data_points]
                    self.ax.plot(timestamps, scores, marker="o", linestyle="-", label=f"学生 {student_id}")

                self.ax.legend()
                self.canvas.draw()
                time.sleep(1)  # 每秒更新一次数据

        threading.Thread(target=update_graph, daemon=True).start()

    def stop_monitoring(self):
        """停止课堂监测"""
        self.is_monitoring = False

def create_ui():
    """创建并启动人脸分析应用的用户界面"""
    root = tk.Tk()
    root.title("人脸分析系统")
    
    # 设置窗口大小和位置
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 900
    window_height = 700
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # 确保目录存在
    os.makedirs("model", exist_ok=True) 
    os.makedirs("data", exist_ok=True)
    
    # 创建应用实例
    try:
        app = FaceAnalysisApp(root, "人脸分析系统")
        return app
    except Exception as e:
        print(f"启动应用时发生错误: {str(e)}")
        messagebox.showerror("启动错误", f"启动应用时发生错误: {str(e)}")
        root.destroy()
        return None

# 如果直接运行app.py，也启动UI
if __name__ == "__main__":
    create_ui()