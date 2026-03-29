import cv2
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication

class CameraWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True

    def run(self):
        try:
            self.log_signal.emit("CameraWorker başladı.")
            self.detector.load_model(self.log_signal.emit)
            self.detector.start_camera()
            while self.running:
                self.msleep(100)
            self.log_signal.emit("CameraWorker durduruldu.")
        except Exception as e:
            self.log_signal.emit(f"Kamera hatası: {str(e)}")
            self.finished_signal.emit()

    def stop(self):
        if self.running:
            self.running = False
            self.detector.stop_camera()
            self.finished_signal.emit()

class CameraDetector:
    def __init__(self, parent, model_path, data_yaml_path):
        self.parent = parent
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = None
        self.cap = None
        self.timer = QTimer(self.parent)
        self.timer.setInterval(30)  # ~30 FPS
        self.timer.timeout.connect(self.update_frame)
        self.parent.log_message("CameraDetector başlatıldı, QTimer bağlandı.")
        # Test sinyali
        self.test_timer = QTimer(self.parent)
        self.test_timer.setInterval(1000)
        self.test_timer.start()
        # QTimer bağlantı testi
        self.timer_test_signal = QTimer(self.parent)
        self.timer_test_signal.setInterval(500)
        self.timer_test_signal.start()

    def load_model(self, callback=None):
        from ultralytics import YOLO
        import torch, os

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path).to(device)
        if callback:
            callback(f"YOLO model loaded on {device}.")

    def start_camera(self):
        if self.model is None:
            self.load_model(self.parent.log_message)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")
        self.parent.log_message(f"Kamera açıldı, çözünürlük: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        # QTimer bağlantısını yeniden kur
        try:
            self.timer.timeout.disconnect()
        except:
            pass
        self.timer.timeout.connect(self.update_frame)
        self.parent.log_message("QTimer bağlantısı yeniden kuruldu.")
        self.timer.start()
        self.parent.log_message("QTimer başlatıldı.")

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.parent.log_message("QTimer durduruldu.")
        if self.test_timer.isActive():
            self.test_timer.stop()
            self.parent.log_message("Test timer durduruldu.")
        if self.timer_test_signal.isActive():
            self.timer_test_signal.stop()
            self.parent.log_message("update_frame QTimer bağlantı testi durduruldu.")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.parent.log_message("📷 Kamera durduruldu.")

    def update_frame(self):
        self.parent.log_message("update_frame çağrıldı.")
        try:
            if not self.cap or not self.cap.isOpened():
                self.parent.log_message("Hata: Kamera bağlantısı kesildi.")
                self.stop_camera()
                self.parent.on_camera_finished()
                return

            ret, frame = self.cap.read()
            if not ret:
                self.parent.log_message("Hata: Frame okunamadı.")
                self.stop_camera()
                self.parent.on_camera_finished()
                return

            self.parent.log_message("Frame okundu, işleniyor...")
            try:
                results = self.model(frame)[0]
                self.parent.log_message(f"YOLO tespiti: {len(results.boxes)} kutu bulundu.")
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                self.parent.log_message(f"YOLO tahmini hatası: {str(e)}")
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.parent.log_message(f"Frame boyutları: {w}x{h}")
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            if qimg.isNull():
                self.parent.log_message("Hata: QImage oluşturulamadı.")
                return
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.parent.camera_label.size(), Qt.KeepAspectRatio)
            if pixmap.isNull():
                self.parent.log_message("Hata: QPixmap oluşturulamadı.")
                return
            self.parent.camera_label.setPixmap(pixmap)
            self.parent.camera_label.update()
            self.parent.log_message(f"Pixmap camera_label'a ayarlandı, boyut: {pixmap.width()}x{pixmap.height()}")
            QApplication.processEvents()
        except Exception as e:
            self.parent.log_message(f"update_frame hatası: {str(e)}")
            self.stop_camera()
            self.parent.on_camera_finished()