import sys
import traceback

def custom_excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = custom_excepthook

# 你的代码


from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import face_recognition
import io

app = FastAPI()

# 静态文件
from fastapi.staticfiles import StaticFiles
app.mount(r"src\static", StaticFiles(directory="static"), name="static")
# 打开摄像头
cap = cv2.VideoCapture(0)

# 已注册人脸库A
registered_faces = {}

def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        for registered_name, registered_encoding in registered_faces.items():
            match = face_recognition.compare_faces([registered_encoding], encoding, tolerance=0.5)
            if match[0]:
                name = registered_name
                break

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_faces(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def home():
    return {"message": "欢迎使用 FastAPI 实时人脸识别"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/register_face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    image = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_encodings = face_recognition.face_encodings(frame)
    if face_encodings:
        registered_faces[name] = face_encodings[0]
        return JSONResponse(content={"message": f"Face {name} registered successfully"}, status_code=200)
    else:
        return JSONResponse(content={"error": "No face detected"}, status_code=400)

@app.get("/list_faces")
def list_faces():
    return JSONResponse(content={"registered_faces": list(registered_faces.keys())})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


