import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class ImageLoader:
    """Handles loading images from a folder and navigating through them."""
    def __init__(self):
        self.root_folder = None
        self.image_paths = []
        self.current_index = 0
        self.valid_trees = [
            'Anhui Barberry', "Beale's barberry", 'Big-fruited Holly', 'camphortree', 'Canadian poplar',
            'castor aralia', 'Chinese cinnamon', 'Chinese horse chestnut', 'Chinese redbud', 'Chinese Toon',
            'Chinese tulip tree', 'Crape myrtle, Crepe myrtle', 'deodar', 'Ford Woodlotus', 'ginkgo, maidenhair tree',
            'Glossy Privet', 'goldenrain tree', 'Japan Arrowwood', 'Japanese cheesewood', 'Japanese Flowering Cherry',
            'Japanese maple', 'Nanmu', 'oleander', 'peach', 'pubescent bamboo', 'southern magnolia',
            'sweet osmanthus', 'tangerine', 'trident maple', 'true indigo', 'wintersweet', 'yew plum pine'
        ]

    def open_folder(self, parent):
        """Opens a folder and loads images."""
        expected_root_folder = os.path.normpath(os.path.join(os.path.expanduser("~"), "Desktop", "Leaves"))
        folder = QFileDialog.getExistingDirectory(
            parent,
            "Klasör Seç (Leaves veya bir alt klasör seçin)",
            os.path.expanduser("~/Desktop")
        )
        if folder:
            normalized_folder = os.path.normpath(folder)
            print(f"Selected folder: {normalized_folder}")
            
            # Seçilen klasörün Leaves veya bir alt klasör olup olmadığını kontrol et
            if normalized_folder == expected_root_folder:
                # Leaves klasörü seçildi, tüm alt klasörleri tara
                self.root_folder = normalized_folder
                parent.log_message(f"'Leaves' klasörü seçildi: {normalized_folder}")
                self.image_paths = []
                subfolders = [
                    d for d in os.listdir(self.root_folder)
                    if os.path.isdir(os.path.join(self.root_folder, d)) and d in self.valid_trees
                ]
                print(f"Detected subfolders: {subfolders}")
                for subfolder in subfolders:
                    folder_path = os.path.join(self.root_folder, subfolder)
                    for file in os.listdir(folder_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(folder_path, file))
            else:
                # Alt klasör seçildi, üst klasörün Leaves olduğunu doğrula
                parent_folder = os.path.dirname(normalized_folder)
                if os.path.normpath(parent_folder) != expected_root_folder:
                    QMessageBox.warning(
                        parent,
                        "Uyarı",
                        f"Yanlış klasör seçildi: {folder}\nLütfen C:/Users/hazan/Desktop/Leaves/ veya bir alt klasörünü seçin."
                    )
                    print(f"Folder selection rejected: {folder}. Expected parent: {expected_root_folder}")
                    self.root_folder = None
                    return []
                self.root_folder = parent_folder
                subfolder_name = os.path.basename(normalized_folder)
                if subfolder_name not in self.valid_trees:
                    QMessageBox.warning(
                        parent,
                        "Uyarı",
                        f"Seçilen klasör ({subfolder_name}) geçerli bir ağaç türü değil: {self.valid_trees}"
                    )
                    print(f"Invalid subfolder: {subfolder_name}")
                    self.root_folder = None
                    return []
                parent.log_message(f"Alt klasör seçildi: {subfolder_name}, root_folder: {self.root_folder}")
                self.image_paths = []
                for file in os.listdir(normalized_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(normalized_folder, file))
                subfolders = [subfolder_name]  # Yalnızca seçilen alt klasörü döndür

            self.image_paths.sort()
            self.current_index = 0
            parent.log_message(f"{len(self.image_paths)} görüntü yüklendi, alt klasörler: {subfolders}")
            return subfolders
        else:
            print("Folder selection canceled.")
            self.root_folder = None
            return []

    def get_current_image_path(self):
        """Returns the path of the current image."""
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def prev_image(self):
        """Navigates to the previous image."""
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            return self.image_paths[self.current_index]
        return None

    def next_image(self):
        """Navigates to the next image."""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            return self.image_paths[self.current_index]
        return None