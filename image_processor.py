import cv2
import numpy as np
from PyQt5.QtGui import QImage

class ImageProcessor:
    """Handles image processing operations."""
    def __init__(self):
        self.original_image = None
        self.result = None

    def apply_filters(self, img_path, filter_settings):
        """Applies selected filters to the image."""
        if not img_path:
            return None, None

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            return None, None
            
        self.original_image = image.copy()
        result = image.copy()
        blurred = cv2.GaussianBlur(result, (5, 5), 0)

        if filter_settings['hsv']:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower = np.array([30, 30, 30])
            upper = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            result = cv2.bitwise_and(result, result, mask=mask)

        if filter_settings['canny']:
            gray = cv2.cvtColor(blurred if not filter_settings['hsv'] else result, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                mask_filled = np.zeros_like(gray)
                cv2.drawContours(mask_filled, [c], -1, 255, -1)
                mask_filled = cv2.GaussianBlur(mask_filled, (5, 5), 0)
                _, mask_filled = cv2.threshold(mask_filled, 127, 255, cv2.THRESH_BINARY)
                result = cv2.bitwise_and(result, result, mask=mask_filled)

        if filter_settings['kmeans']:
            Z = blurred.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center_intensities = np.mean(centers, axis=1)
            background_idx = np.argmax(center_intensities)
            mask = (labels != background_idx).astype(np.uint8) * 255
            mask = mask.reshape(image.shape[:2])
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            result = cv2.bitwise_and(result, result, mask=mask)

        if filter_settings['threshold']:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            result = cv2.bitwise_and(result, result, mask=mask)

        if filter_settings['transparent']:
            if result.shape[2] == 4:
                mask = result[:, :, 3]
            else:
                mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            b, g, r = cv2.split(result[:, :, :3] if result.shape[2] == 4 else result)
            result = cv2.merge((b, g, r, mask))

        if filter_settings['crop']:
            if result.shape[2] == 4:
                mask = result[:, :, 3]
            else:
                mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                result = result[y:y + h, x:x + w]

        self.result = result
        return self.original_image, result

    def convert_to_qimage(self, image):
        """Converts an OpenCV image to QImage format."""
        if image is None:
            return None

        if len(image.shape) == 2:
            result_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            h, w = result_disp.shape[:2]
            ch = 3
            format = QImage.Format_RGB888
            data = result_disp.data
        elif image.shape[2] == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            h, w = rgba.shape[:2]
            ch = 4
            format = QImage.Format_RGBA8888
            data = rgba.data
        else:
            rgb_result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb_result.shape[:2]
            ch = 3
            format = QImage.Format_RGB888
            data = rgb_result.data

        bytes_per_line = ch * w
        return QImage(data, w, h, bytes_per_line, format)