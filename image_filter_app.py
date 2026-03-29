import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import cv2
import time
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from ui_components import UIComponents
from image_loader import ImageLoader
from image_processor import ImageProcessor
from yolo_labeler import YoloLabeler
from model_trainer import ModelTrainer
from camera_detector import CameraDetector


class CameraWorker(QThread):
    frame_signal = pyqtSignal(object)
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
            self.detector.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.detector.cap.isOpened():
                self.log_signal.emit("Hata: Kamera açılamadı.")
                self.finished_signal.emit()
                return
            # Kamera çözünürlüğünü artır
            self.detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.log_signal.emit(f"Kamera açıldı, çözünürlük: {self.detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            while self.running:
                ret, frame = self.detector.cap.read()
                if ret:
                    self.frame_signal.emit(frame)
                else:
                    self.log_signal.emit("Hata: Frame okunamadı.")
                    break
                self.msleep(50)  # 20 FPS (~50ms)
            self.log_signal.emit("CameraWorker durduruldu.")
        except Exception as e:
            self.log_signal.emit(f"Kamera hatası: {str(e)}")
            self.finished_signal.emit()

    def stop(self):
        self.running = False
        if self.detector.cap and self.detector.cap.isOpened():
            self.detector.cap.release()
        self.finished_signal.emit()


class ImageFilterApp(QWidget):
    """Main application class for the image filter GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görüntü İşleme ve YOLO Eğitimi Arayüzü")
        self.setMinimumSize(1000, 700)
        self.ui = UIComponents()
        self.loader = ImageLoader()
        self.processor = ImageProcessor()
        self.labeler = YoloLabeler()
        self.trainer = ModelTrainer()
        self.detector = CameraDetector(
            parent=self,
            model_path=r"C:\Users\hazan\Desktop\camera_leaf\runs\detect\train\weights\best.pt",
            data_yaml_path=r"C:\Users\hazan\Desktop\dataset\data.yaml"
        )
        self.training_thread = None
        self.camera_thread = None
        self.camera_running = False
        self.frame_count = 0  # Logları seyreltmek için sayaç
        self.last_detected_objects = []  # Son tespitleri önbelleğe al
        self.last_detection_time = 0  # Son tespit zamanı
        self.init_ui()

    def init_ui(self):
        self.setLayout(self.ui.setup_layout())
        self.ui.open_button.clicked.connect(self.open_folder)
        self.ui.prev_button.clicked.connect(self.prev_image)
        self.ui.next_button.clicked.connect(self.next_image)
        self.ui.apply_button.clicked.connect(self.apply_filters)
        self.ui.label_button.clicked.connect(self.label_all_images)
        self.ui.train_button.clicked.connect(self.start_training)
        self.ui.camera_button.clicked.connect(self.start_camera_detection)
        self.ui.stop_camera_button.clicked.connect(self.stop_camera_detection)
        self.ui.tree_combo.currentTextChanged.connect(self.update_class_id)
        self.ui.camera_label.setVisible(True)
        self.log_message(f"camera_label boyutları: {self.ui.camera_label.size().width()}x{self.ui.camera_label.size().height()}")
        self.log_message(f"camera_label görünür mü: {self.ui.camera_label.isVisible()}")
        # Test görüntüsü
        test_image_path = r"C:\Users\hazan\Desktop\test.jpg"
        if os.path.exists(test_image_path):
            pixmap = QPixmap(test_image_path).scaled(400, 300, Qt.KeepAspectRatio)
            if not pixmap.isNull():
                self.ui.camera_label.setPixmap(pixmap)
                self.ui.camera_label.update()
                self.log_message("Test görüntüsü camera_label'a ayarlandı.")
                QApplication.processEvents()
            else:
                self.log_message("Hata: Test görüntüsü QPixmap oluşturulamadı.")
        else:
            self.log_message(f"Hata: Test görüntüsü bulunamadı: {test_image_path}")

    def log_message(self, message):
        # Logları hem GUI'ye hem dosyaya yaz
        with open("app.log", "a") as f:
            f.write(f"{message}\n")
        self.ui.log_textedit.append(message)
        self.ui.log_textedit.verticalScrollBar().setValue(
            self.ui.log_textedit.verticalScrollBar().maximum()
        )

    def update_class_id(self, text):
        self.labeler.update_class_id(text, self.loader.root_folder)

    def open_folder(self):
        subfolders = self.loader.open_folder(self)
        self.ui.tree_combo.clear()
        for subfolder in sorted(subfolders):
            self.ui.tree_combo.addItem(subfolder, subfolder)
        if self.loader.image_paths:
            current_folder = os.path.basename(os.path.dirname(self.loader.image_paths[self.loader.current_index]))
            if subfolders and current_folder in subfolders:
                self.ui.tree_combo.setCurrentText(current_folder)
        self.labeler.update_class_id(current_folder, self.loader.root_folder)
        self.show_image()
        self.log_message(f"📁 Klasör yüklendi: {len(self.loader.image_paths)} görüntü bulundu")

    def show_image(self):
        img_path = self.loader.get_current_image_path()
        if img_path:
            pixmap = QPixmap(img_path)
            self.ui.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            current_folder = os.path.basename(os.path.dirname(img_path))
            if self.ui.tree_combo.count() > 0:
                self.ui.tree_combo.setCurrentText(current_folder)
            self.labeler.update_class_id(current_folder, self.loader.root_folder)

    def prev_image(self):
        img_path = self.loader.prev_image()
        if img_path:
            self.show_image()

    def next_image(self):
        img_path = self.loader.next_image()
        if img_path:
            self.show_image()

    def apply_filters(self):
        filter_settings = {
            'hsv': self.ui.hsv_checkbox.isChecked(),
            'canny': self.ui.canny_checkbox.isChecked(),
            'kmeans': self.ui.kmeans_checkbox.isChecked(),
            'threshold': self.ui.threshold_checkbox.isChecked(),
            'crop': self.ui.crop_checkbox.isChecked(),
            'transparent': self.ui.transparent_checkbox.isChecked()
        }
        original, result = self.processor.apply_filters(self.loader.get_current_image_path(), filter_settings)
        if original is not None:
            qimg_original = self.processor.convert_to_qimage(original)
            self.ui.image_label.setPixmap(QPixmap.fromImage(qimg_original).scaled(300, 300, Qt.KeepAspectRatio))
        if result is not None:
            qimg_result = self.processor.convert_to_qimage(result)
            self.ui.result_label.setPixmap(QPixmap.fromImage(qimg_result).scaled(300, 300, Qt.KeepAspectRatio))
        self.labeler.create_yolo_label(result, self.loader.get_current_image_path(), self.labeler.selected_tree)
        self.log_message("🎨 Filtreler uygulandı ve YOLO etiketi oluşturuldu")

    def label_all_images(self):
        if not self.labeler.selected_tree:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir ağaç türü seçin!")
            return
        filter_settings = {
            'hsv': self.ui.hsv_checkbox.isChecked(),
            'canny': self.ui.canny_checkbox.isChecked(),
            'kmeans': self.ui.kmeans_checkbox.isChecked(),
            'threshold': self.ui.threshold_checkbox.isChecked(),
            'crop': self.ui.crop_checkbox.isChecked(),
            'transparent': self.ui.transparent_checkbox.isChecked()
        }
        self.log_message(f"🏷️ Tüm görüntüler etiketleniyor: {self.labeler.selected_tree}")
        labeled_images = self.labeler.label_all_images(self.loader.image_paths, self.labeler.selected_tree, self.loader.root_folder, filter_settings)
        self.log_message(f"✅ {labeled_images} görüntü başarıyla etiketlendi!")

    def start_training(self):
        pass

    def start_camera_detection(self):
        if self.camera_running:
            self.log_message("Kamera zaten çalışıyor.")
            return
        try:
            self.camera_running = True
            self.ui.camera_button.setEnabled(False)
            self.ui.camera_button.setText("Kamera Tespiti Devam Ediyor...")
            self.ui.stop_camera_button.setEnabled(True)
            self.ui.camera_label.setVisible(True)

            self.camera_worker = CameraWorker(self.detector)
            self.camera_worker.frame_signal.connect(self.update_camera_label)
            self.camera_worker.log_signal.connect(self.log_message)
            self.camera_worker.finished_signal.connect(self.on_camera_finished)
            self.camera_worker.start()

            self.log_message("📷 Kamera başlatıldı.")
        except Exception as e:
            self.camera_running = False
            self.log_message(f"Kamera başlatma hatası: {str(e)}")
            QMessageBox.warning(self, "Uyarı", f"Kamera başlatılamadı: {str(e)}")
            self.ui.camera_button.setEnabled(True)
            self.ui.camera_button.setText("Kamera Tespitini Başlat")
            self.ui.stop_camera_button.setEnabled(False)

    def stop_camera_detection(self):
        if self.camera_running and hasattr(self, 'camera_worker') and self.camera_worker:
            self.camera_worker.stop()
            self.camera_running = False
        self.ui.camera_label.setVisible(True)
        self.log_message(f"camera_label görünür mü (stop_camera): {self.ui.camera_label.isVisible()}")

    def on_camera_finished(self):
        if self.camera_running:
            self.detector.stop_camera()
            self.camera_running = False
            self.ui.camera_button.setEnabled(True)
            self.ui.camera_button.setText("Kamera Tespitini Başlat")
            self.ui.stop_camera_button.setEnabled(False)
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.quit()
                self.camera_thread.wait()
            self.ui.camera_label.setVisible(True)
            self.log_message(f"camera_label görünür mü (on_camera_finished): {self.ui.camera_label.isVisible()}")
            self.log_message("📷 Kamera durduruldu.")

    def update_camera_label(self, frame):
        try:
            # Her 30 frame'de bir genel durum logu
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.log_message("update_camera_label çağrıldı.")
            
            # Frame'in geçerli olduğunu kontrol et
            if frame is None or frame.size == 0:
                self.log_message("Hata: Geçersiz veya boş frame alındı.")
                return
            if self.frame_count % 30 == 0:
                self.log_message(f"Frame şekli: {frame.shape}")

            # YOLO modelinin yüklü olduğundan emin ol
            if self.detector.model is None:
                self.log_message("Hata: YOLO modeli yüklenmedi, yükleniyor...")
                self.detector.load_model(self.log_message)
            
            # YOLO ile nesne tespiti
            results = self.detector.model(frame)[0]
            if self.frame_count % 30 == 0:
                self.log_message(f"YOLO sonuçları alındı, kutu sayısı: {len(results.boxes)}")
            
            detected_objects = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.detector.model.names[cls]
                conf = float(box.conf[0])
                if self.frame_count % 30 == 0:
                    self.log_message(f"Ham tespit: {label} (Güven: {conf:.2f})")
                if conf > 0.05:  # Güven skoru eşiğini düşürdüm
                    detected_objects.append((label, conf, (x1, y1, x2, y2)))

            # Tespit edilen yaprak türlerini log et ve önbelleğe al
            current_time = time.time()
            if detected_objects:
                self.last_detected_objects = detected_objects  # Son tespitleri önbelleğe al
                self.last_detection_time = current_time  # Tespit zamanını güncelle
                for label, conf, _ in detected_objects:
                    self.log_message(f"Yaprak tespit edildi: {label} (Güven: {conf:.2f})")
            else:
                # Önbellekteki son tespitleri 2 saniye boyunca kullan
                if self.last_detected_objects and (current_time - self.last_detection_time) < 2:
                    for label, conf, (x1, y1, x2, y2) in self.last_detected_objects:
                        self.log_message(f"Son tespit (önbellek): {label} (Güven: {conf:.2f})")
                        # Önbellekteki bounding box'ı çiz
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    self.log_message("Hiçbir yaprak tespit edilmedi.")

            # Yeni tespitler için bounding box ve etiket çiz
            for label, conf, (x1, y1, x2, y2) in detected_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Frame'i GUI'de göster
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            if self.frame_count % 30 == 0:
                self.log_message(f"RGB frame boyutları: {w}x{h}")
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            if qimg.isNull():
                self.log_message("Hata: QImage oluşturulamadı.")
                return
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.ui.camera_label.size(), Qt.KeepAspectRatio
            )
            if pixmap.isNull():
                self.log_message("Hata: QPixmap oluşturulamadı.")
                return
            self.ui.camera_label.setPixmap(pixmap)
            self.ui.camera_label.repaint()
            if self.frame_count % 30 == 0:
                self.log_message("Frame GUI'de gösterildi.")
        except Exception as e:
            self.log_message(f"update_camera_label hatası: {str(e)}")

    def __del__(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.stop_camera()


def main():
    """Ana uygulama fonksiyonu."""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Görüntü İşleme ve YOLO Eğitimi")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("Görüntü İşleme Ekibi")
        
        window = ImageFilterApp()
        window.show()
        
        print("Uygulama başarıyla başlatıldı!")
        print("Klasör seçin ve görüntü işlemeye başlayın!")
        
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Uygulama başlatma hatası: {str(e)}")
        if 'app' in locals():
            QMessageBox.critical(None, "Başlatma Hatası", f"Uygulama başlatılamadı: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()