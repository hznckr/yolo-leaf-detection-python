import os
import shutil
import yaml
import re
from ultralytics import YOLO
import threading
import time
import torch

class ModelTrainer:
    """Handles YOLOv8 training and logs."""

    def __init__(self):
        self.model = None
        self.training_thread = None
        self.is_training = False

    def clean_filename(self, filename):
        """Removes special characters and spaces from filenames."""
        clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        clean_name = re.sub(r'_+', '_', clean_name)
        return clean_name.strip('_')

    def prepare_dataset(self, root_folder):
        """Prepares dataset with 80/20 train/val split, only including images with labels."""
        dataset_path = os.path.join(os.path.expanduser("~"), "Desktop", "dataset")
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(os.path.join(dataset_path, "images", "train"))
        os.makedirs(os.path.join(dataset_path, "images", "val"))
        os.makedirs(os.path.join(dataset_path, "labels", "train"))
        os.makedirs(os.path.join(dataset_path, "labels", "val"))

        valid_trees = [
            'Anhui Barberry', "Beale's barberry", 'Big-fruited Holly', 'camphortree', 'Canadian poplar',
            'castor aralia', 'Chinese cinnamon', 'Chinese horse chestnut', 'Chinese redbud', 'Chinese Toon',
            'Chinese tulip tree', 'Crape myrtle, Crepe myrtle', 'deodar', 'Ford Woodlotus', 'ginkgo, maidenhair tree',
            'Glossy Privet', 'goldenrain tree', 'Japan Arrowwood', 'Japanese cheesewood', 'Japanese Flowering Cherry',
            'Japanese maple', 'Nanmu', 'oleander', 'peach', 'pubescent bamboo', 'southern magnolia',
            'sweet osmanthus', 'tangerine', 'trident maple', 'true indigo', 'wintersweet', 'yew plum pine'
        ]
        all_trees = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d)) and d in valid_trees]
        valid_image_count = 0
        skipped_files = []
        copied_files = []

        for tree in all_trees:
            tree_folder = os.path.join(root_folder, tree)
            desktop_labels_path = os.path.join(os.path.expanduser("~"), "Desktop", tree)
            
            if not os.path.exists(desktop_labels_path):
                skipped_files.append(f"Label folder missing for {tree}: {desktop_labels_path}")
                continue

            image_files = [f for f in os.listdir(tree_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images = len(image_files)
            train_split = int(total_images * 0.8) if total_images > 0 else 0

            for idx, img_file in enumerate(sorted(image_files)):
                src_img = os.path.join(tree_folder, img_file)
                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label = os.path.join(desktop_labels_path, label_file)

                if os.path.exists(src_label):
                    with open(src_label, 'r') as f:
                        label_content = f.read().strip()
                        if not label_content:
                            skipped_files.append(f"Skipping {img_file} in {tree}: Empty label file at {src_label}")
                            continue

                    clean_img_file = self.clean_filename(f"{tree}_{img_file}")
                    clean_label_file = self.clean_filename(f"{tree}_{os.path.splitext(img_file)[0]}.txt")

                    if idx < train_split:
                        dst_img = os.path.join(dataset_path, "images", "train", clean_img_file)
                        dst_label = os.path.join(dataset_path, "labels", "train", clean_label_file)
                    else:
                        dst_img = os.path.join(dataset_path, "images", "val", clean_img_file)
                        dst_label = os.path.join(dataset_path, "labels", "val", clean_label_file)

                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_label, dst_label)
                    valid_image_count += 1
                    copied_files.append(f"Copied: {dst_img} and {dst_label}")
                else:
                    skipped_files.append(f"Skipping {img_file} in {tree}: No label file found at {src_label}")

        if copied_files:
            print("Copied files:")
            for copied in copied_files:
                print(copied)
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files due to missing or empty labels:")
            for skipped in skipped_files:
                print(skipped)

        print(f"Prepared dataset with {valid_image_count} valid images.")
        if valid_image_count == 0:
            raise ValueError("No valid images with labels found. Please run 'Label All' for each tree type in C:/Users/hazan/Desktop/Leaves/. Ensure label files exist in C:/Users/hazan/Desktop/<tree_name>/.")
        return dataset_path

    def create_data_yaml(self, dataset_path, all_trees):
        """Creates data.yaml for YOLOv8."""
        fixed_classes = sorted(all_trees)
        data = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(fixed_classes),
            'names': fixed_classes
        }
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
        print(f"Created data.yaml at {yaml_path} with {len(fixed_classes)} classes: {fixed_classes}")
        return yaml_path

    def log_training_progress(self, callback, log_file_path):
        """Monitors training log file and sends updates to UI."""
        if not callback:
            return
            
        last_position = 0
        while self.is_training:
            try:
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        if new_content:
                            lines = new_content.strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    if 'Epoch' in line and 'GPU_mem' in line:
                                        callback(f" {line.strip()}")
                                    elif 'Class' in line and 'mAP50' in line:
                                        callback(f" {line.strip()}")
                                    elif 'all' in line and any(x in line for x in ['mAP50', 'Precision', 'Recall']):
                                        callback(f" {line.strip()}")
                                    elif 'EarlyStopping' in line:
                                        callback(f" {line.strip()}")
                                    elif 'epochs completed' in line:
                                        callback(f" {line.strip()}")
                                    elif 'Results saved to' in line:
                                        callback(f" {line.strip()}")
                            last_position = f.tell()
                time.sleep(1)
            except Exception as e:
                print(f"Log monitoring error: {e}")
                time.sleep(2)

    def train_model(self, data_yaml_path, epochs=50, batch_size=8, resume_from_previous=False, callback=None):
        """Trains YOLOv8 model and calls callback for logging."""
        try:
            if callback:
                callback(" YOLO Eğitimi Başlatılıyor...")
                callback(f" Parametreler: {epochs} epoch, batch boyutu: {batch_size}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if callback:
                callback(f" Çalışma cihazı: {device}")
            if device == "cuda" and callback:
                callback(f" GPU adı: {torch.cuda.get_device_name(0)}")
                callback(f" Toplam GPU Belleği (MiB): {torch.cuda.get_device_properties(0).total_memory // (1024**2)}")
                callback(f" cuDNN sürümü: {torch.backends.cudnn.version()}")

            self.model = YOLO("yolov8n.pt").to(device)

            if callback:
                callback(" YOLO modeli yüklendi")
                callback(" Eğitim başlıyor... (Bu işlem uzun sürebilir)")

            self.is_training = True

            if callback:
                log_thread = threading.Thread(
                    target=self.log_training_progress,
                    args=(callback, None),
                    daemon=True
                )
                log_thread.start()

            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                workers=2,
                verbose=True,
                patience=10,
                save=True,
                plots=True,
                project="runs/detect",
                resume=resume_from_previous
            )

            self.is_training = False

            if callback:
                callback(" Eğitim tamamlandı!")
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                    if 'metrics/mAP50(B)' in metrics:
                        callback(f" Final mAP50: {metrics['metrics/mAP50(B)']:.3f}")
                    if 'metrics/mAP50-95(B)' in metrics:
                        callback(f" Final mAP50-95: {metrics['metrics/mAP50-95(B)']:.3f}")

                save_dir = getattr(self.model.trainer, 'save_dir', 'runs/detect/train*')
                callback(f" Model kaydedildi: {save_dir}")

                callback("=" * 50)
                callback(" EĞİTİM SONUÇLARI:")
                callback(" Yüksek mAP50 (>0.95) = Mükemmel performans")
                callback(" Yüksek mAP50-95 (>0.80) = Çok iyi genel performans")
                callback(" Early Stopping kullanıldı = Overfitting önlendi")
                callback("=" * 50)

            return results

        except Exception as e:
            self.is_training = False
            error_msg = f" Eğitim hatası: {str(e)}"
            print(error_msg)
            if callback:
                callback(error_msg)
                callback(" İpucu: Batch boyutunu küçültmeyi deneyin")
                callback(" İpucu: GPU belleğinizi kontrol edin")
                callback(" İpucu: C:\\Users\\hazan\\Desktop\\dataset\\ klasörünü ve data.yaml dosyasını kontrol edin")
                callback(" İpucu: Tüm ağaç türleri için 'Label All' ile etiket oluşturun")
            raise e