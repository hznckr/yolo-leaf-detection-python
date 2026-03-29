# 🍃 Auto-Labeler & Tree Identification System with YOLOv8

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-41CD52?style=for-the-badge&logo=qt&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)

An advanced desktop application that automates the labeling process for YOLOv8 datasets and performs real-time tree species identification. This tool combines traditional computer vision filters with deep learning for a seamless "image-to-model" workflow.

## 🚀 Key Features
- **Smart Auto-Labeling:** Automatically generates YOLO-format segmentation labels using contour detection and image processing.
- **Advanced Pre-processing:** Interactive UI to apply HSV Filtering, Canny Edge Detection, K-Means Segmentation, and Thresholding.
- **Unified Dataset Management:** Automatically organizes images and labels into a training-ready folder structure.
- **Real-time Detection:** Live camera inference support for immediate model testing.
- **PyQt5 GUI:** A professional user interface for managing datasets, training, and testing in one place.

## 🛠 Tech Stack
- **GUI:** PyQt5
- **Image Processing:** OpenCV, NumPy
- **Deep Learning:** Ultralytics YOLOv8, PyTorch
- **Automation:** Shutil, OS (File management automation)

## 📖 How It Works
1. **Data Preperation:** Select a folder containing tree leaf images.
2. **Filtering:** Apply filters (HSV, Canny, etc.) to isolate the leaf from the background.
3. **Auto-Label:** Click "Label All" to generate normalized YOLO coordinates automatically based on the largest contours.
4. **Train:** Start YOLOv8 training directly from the UI with the "Start Training" button.
5. **Test:** Use "Start Camera Detection" to see your trained model in action.

## ⚙️ Installation
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/yolo-leaf-detection-python.git](https://github.com/YOUR_USERNAME/yolo-leaf-detection-python.git)

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
