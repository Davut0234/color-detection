import cv2
import time
import threading
import math
from ultralytics import YOLO

class Camera:
    def __init__(self, modelFile, videoCapturePort, logFileName='log.txt'):
        self.modelFile = modelFile
        self.videoCapturePort = videoCapturePort
        self.model = YOLO(self.modelFile)
        self.logFileName = logFileName
        self.tracked_objects = {}
        self.camera = self.openCamera()
        self.currentFrame = None
        self.isActive = True
        self.isCameraOpened = False
        self.cameraThread = threading.Thread(target=self.readCamera)

        # FPS (Kare Per Saniye) sayacı ekleyin
        self.fps = 0
        self.start_time = time.time()

        self.cameraThread.start()
        while not self.isCameraOpened:
            self.isCameraOpened = self.camera.isOpened()

    def openCamera(self):
        return cv2.VideoCapture(self.videoCapturePort)

    def readCamera(self):
        while self.isActive:
            ret, frame = self.camera.read()
            self.currentFrame = frame

        self.camera.release()

    def getCurrentFrame(self):
        return self.currentFrame

    def detectObjects(self, objectsToDetect, frame):
        results = self.model(frame, stream=True)
        return results

    def track_objects(self, frame, positions):
        frame_height, frame_width, _ = frame.shape
        center_x, center_y = frame_width // 2, frame_height // 2

        for r in positions:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100

                if confidence > 0.8:
                    # Nesnenin rengini tespit et
                    detected_color = self.detectColor(frame[y1:y2, x1:x2])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2

                    shifted_centroid_x = centroid_x - center_x
                    shifted_centroid_y = center_y - centroid_y

                    x, y = shifted_centroid_x, shifted_centroid_y

                    text = f"X: {x:.2f}, Y: {y:.2f}, Color: {detected_color}"
                    with open(self.logFileName, "a") as file:
                        file.write(text + "\n")
                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    color = (255, 255, 255)
                    thickness = 1
                    cv2.putText(frame, text, org, font, fontScale, color, thickness)

        return frame

    def detectColor(self, roi):
        # ROI (Region of Interest) içindeki ortalama rengi tespit et
        average_color = cv2.mean(roi)[:3]

        # En yakın renk adını al
        detected_color = self.get_closest_color_name(average_color)

        return detected_color

    def get_closest_color_name(self, rgb_color):
        # Burada, önceden tanımlanmış renklerle bir renk eşleme algoritması kullanılabilir
        # Bu örnekte, basit bir RGB renk eşleme yöntemi kullanılmıştır.
        colors = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            # Diğer renkleri ekleyebilirsiniz
        }

        min_distance = float('inf')
        closest_color = None

        for color_name, color_value in colors.items():
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_color, color_value)))

            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color

# Usage
cameraProcess = Camera('./best.pt', 0)

while True:
    if cameraProcess.isCameraOpened:
        try:
            currentFrame = cameraProcess.getCurrentFrame()
            currentFrame = cv2.resize(currentFrame, (640, 640))

            objectsToDetect = ['mouse']
            positions = cameraProcess.detectObjects(objectsToDetect, currentFrame)

            currentFrame = cameraProcess.track_objects(currentFrame, positions)

            # Nesne tespiti ve takibi sonuçlarını göster
            cv2.imshow("Object Detection and Tracking", currentFrame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cameraProcess.isActive = False
                cameraProcess.cameraThread.join()
                break
            time.sleep(0.03)
        except Exception as e:
            print(e)
            time.sleep(3)

cv2.destroyAllWindows()
