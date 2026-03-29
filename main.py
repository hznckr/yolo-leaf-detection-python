#!/usr/bin/env python3
"""
Ana uygulama giriş noktası - OpenMP sorunu çözümlü
"""

# OpenMP çakışması çözümü - MUTLAKA EN BAŞTA OLMALI
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from image_filter_app import ImageFilterApp
from image_loader import ImageLoader
from image_processor import ImageProcessor
from yolo_labeler import YoloLabeler
from model_trainer import ModelTrainer
from ui_components import UIComponents
from train_page import TrainPage


def main():
    """Ana uygulama fonksiyonu."""
    try:
        app = QApplication(sys.argv)
        
        # Uygulama özelliklerini ayarla
        app.setApplicationName("Görüntü İşleme ve YOLO Eğitimi")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("Görüntü İşleme Ekibi")
        
        # Ana pencereyi oluştur ve göster
        window = ImageFilterApp()
        window.show()
        
        print(" Uygulama başarıyla başlatıldı!")
        print(" Klasör seçin ve görüntü işlemeye başlayın!")
        
        # Olay döngüsünü başlat
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f" Uygulama başlatma hatası: {str(e)}")
        if 'app' in locals():
            QMessageBox.critical(None, "Başlatma Hatası", f"Uygulama başlatılamadı:\
{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()