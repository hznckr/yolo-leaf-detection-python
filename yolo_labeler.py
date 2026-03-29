import os
import cv2
import numpy as np
from image_processor import ImageProcessor
import shutil
import time

class YoloLabeler:
    """Handles YOLO label creation and saving."""
    def __init__(self):
        self.class_id = 0
        self.selected_tree = None
        self.nc = 0  # toplam sınıf sayısı

    def update_class_id(self, text, root_folder):
        """Updates the class ID based on the selected tree."""
        if text and root_folder:
            self.selected_tree = text
            subfolders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
            subfolders.sort()  # alfabetik sıralama ile tutarlılık
            self.class_id = subfolders.index(text) if text in subfolders else 0
            self.nc = len(subfolders)
            print(f"Selected tree: {text}, Class ID: {self.class_id}, Total classes: {self.nc}")

    def create_yolo_label(self, image, img_path, selected_tree):
        """Creates a YOLO label for a single image and adds to unified dataset."""
        if image is None or not selected_tree:
            print("No image or tree selected.")
            return

        try:
            if image.shape[2] == 4:
                mask = image[:, :, 3]
            else:
                mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No contours found.")
                return

            largest_contour = max(contours, key=cv2.contourArea)
            h, w = mask.shape
            normalized_points = largest_contour.reshape(-1, 2).astype(np.float32) / [w, h]
            points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])
            yolo_label = f"{self.class_id} {points_str}"

            # Save to Desktop/<selected_tree> folder
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", selected_tree)
            if not os.path.exists(desktop_path):
                os.makedirs(desktop_path)
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(desktop_path, f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                f.write(yolo_label + '\n')

            print(f"YOLO label saved to {txt_path}")

            # Add to unified dataset
            dataset_path = os.path.join(os.path.expanduser("~"), "Desktop", "dataset")
            train_img_path = os.path.join(dataset_path, "images", "train")
            train_label_path = os.path.join(dataset_path, "labels", "train")
            
            if not os.path.exists(train_img_path):
                os.makedirs(train_img_path)
            if not os.path.exists(train_label_path):
                os.makedirs(train_label_path)
            
            # Avoid name conflicts with timestamp
            timestamp = int(time.time())
            new_img_name = f"{selected_tree}_{base_name}_{timestamp}.jpg"
            new_label_name = f"{selected_tree}_{base_name}_{timestamp}.txt"
            
            shutil.copy(img_path, os.path.join(train_img_path, new_img_name))
            shutil.copy(txt_path, os.path.join(train_label_path, new_label_name))
            print(f"Added to dataset: {new_img_name}, {new_label_name}")

        except Exception as e:
            print(f"Error creating YOLO label: {str(e)}")

    def label_all_images(self, image_paths, selected_tree, root_folder, filter_settings):
        """Labels all images in the selected tree folder and adds to unified dataset."""
        if not image_paths or not selected_tree:
            print("No images or tree selected.")
            return

        tree_folder = os.path.join(root_folder, selected_tree)
        if not os.path.exists(tree_folder):
            print(f"Folder {tree_folder} not found.")
            return

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", selected_tree)
        dataset_path = os.path.join(os.path.expanduser("~"), "Desktop", "dataset")
        train_img_path = os.path.join(dataset_path, "images", "train")
        train_label_path = os.path.join(dataset_path, "labels", "train")
        
        if not os.path.exists(desktop_path):
            os.makedirs(desktop_path)
        if not os.path.exists(train_img_path):
            os.makedirs(train_img_path)
        if not os.path.exists(train_label_path):
            os.makedirs(train_label_path)
        
        processor = ImageProcessor()
        processed_count = 0
        
        for img_path in [p for p in image_paths if selected_tree in os.path.basename(os.path.dirname(p))]:
            try:
                original, result = processor.apply_filters(img_path, filter_settings)
                if result is None:
                    print(f"Failed to process image: {img_path}")
                    continue

                if result.shape[2] == 4:
                    mask = result[:, :, 3]
                else:
                    mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print(f"No contours found for {img_path}")
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                h, w = mask.shape
                normalized_points = largest_contour.reshape(-1, 2).astype(np.float32) / [w, h]
                points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])
                yolo_label = f"{self.class_id} {points_str}"

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(desktop_path, f"{base_name}.txt")
                with open(txt_path, 'w') as f:
                    f.write(yolo_label + '\n')

                # Add to unified dataset
                timestamp = int(time.time())
                new_img_name = f"{selected_tree}_{base_name}_{timestamp}.jpg"
                new_label_name = f"{selected_tree}_{base_name}_{timestamp}.txt"
                shutil.copy(img_path, os.path.join(train_img_path, new_img_name))
                shutil.copy(txt_path, os.path.join(train_label_path, new_label_name))
                print(f"Added to dataset: {new_img_name}, {new_label_name}")

                processed_count += 1
                print(f"YOLO label saved to {txt_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        print(f"All images labeled for {selected_tree}. Processed: {processed_count} images.")