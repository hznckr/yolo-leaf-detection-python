from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class UIComponents:
    """Handles creation and setup of UI components."""
    def __init__(self):
        # Labels
        self.camera_label = QLabel("Camera Output")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid gray;")
        self.camera_label.setMinimumSize(400, 300)
        self.camera_label.setVisible(True)  # Görünürlüğü zorla

        self.original_label = QLabel("Original Image")
        self.original_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setMinimumSize(300, 300)

        self.result_label_widget = QLabel("Filtered Image")
        self.result_label_widget.setFont(QFont("Arial", 10, QFont.Bold))
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid gray;")
        self.result_label.setMinimumSize(300, 300)

        # Checkboxes
        self.hsv_checkbox = QCheckBox("HSV Filter")
        self.canny_checkbox = QCheckBox("Canny Edge")
        self.kmeans_checkbox = QCheckBox("K-Means Segmentation")
        self.threshold_checkbox = QCheckBox("Threshold")
        self.crop_checkbox = QCheckBox("Crop")
        self.transparent_checkbox = QCheckBox("Transparent Background")
        self.resume_training_checkbox = QCheckBox("Resume Previous Training")

        # Buttons
        self.open_button = QPushButton("Select Folder")
        self.prev_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")
        self.apply_button = QPushButton("Apply Filter")
        self.label_button = QPushButton("Label All")
        self.train_button = QPushButton("Start YOLO Training")
        self.camera_button = QPushButton("Start Camera Detection")
        self.stop_camera_button = QPushButton("Stop Camera Detection")
        self.stop_camera_button.setEnabled(False)  # Başlangıçta devre dışı

        # Style buttons
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        camera_button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """
        stop_button_style = """
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """

        for button in [self.open_button, self.prev_button, self.next_button,
                       self.apply_button, self.label_button, self.train_button]:
            button.setStyleSheet(button_style)
        self.camera_button.setStyleSheet(camera_button_style)
        self.stop_camera_button.setStyleSheet(stop_button_style)

        # ComboBox & Log
        self.tree_combo = QComboBox()
        self.tree_combo.setMinimumWidth(200)
        self.log_textedit = QTextEdit()
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setMaximumHeight(150)
        self.log_textedit.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ccc;")

    def setup_layout(self):
        """Sets up the layout for UI components."""
        # Filter checkboxes
        filter_box = QHBoxLayout()
        for cb in [self.hsv_checkbox, self.canny_checkbox, self.kmeans_checkbox,
                   self.threshold_checkbox, self.crop_checkbox, self.transparent_checkbox]:
            filter_box.addWidget(cb)

        # Camera display
        self.camera_vbox = QVBoxLayout()
        self.camera_vbox.addWidget(QLabel("Camera Detection"))
        self.camera_vbox.addWidget(self.camera_label)
        self.camera_vbox.addStretch()  # Layout'u dengelemek için

        # Image display
        self.original_vbox = QVBoxLayout()
        self.original_vbox.addWidget(self.original_label)
        self.original_vbox.addWidget(self.image_label)

        self.result_vbox = QVBoxLayout()
        self.result_vbox.addWidget(self.result_label_widget)
        self.result_label.setVisible(True)  # Görünürlüğü zorla
        self.result_vbox.addWidget(self.result_label)

        self.img_box = QHBoxLayout()
        self.img_box.addLayout(self.original_vbox)
        self.img_box.addLayout(self.result_vbox)
        self.img_box.addLayout(self.camera_vbox)
        self.img_box.addStretch()  # Layout'u dengelemek için

        # Control buttons
        button_box = QHBoxLayout()
        for btn in [self.open_button, self.prev_button, self.next_button,
                    self.apply_button, self.label_button, self.train_button,
                    self.camera_button, self.stop_camera_button]:
            button_box.addWidget(btn)

        # Tree selection
        tree_box = QHBoxLayout()
        tree_label = QLabel("Select Tree:")
        tree_label.setFont(QFont("Arial", 9, QFont.Bold))
        tree_box.addWidget(tree_label)
        tree_box.addWidget(self.tree_combo)
        tree_box.addWidget(self.resume_training_checkbox)
        tree_box.addStretch()

        # Main layout
        main_vbox = QVBoxLayout()
        main_vbox.addLayout(self.img_box)
        main_vbox.addLayout(filter_box)
        main_vbox.addLayout(button_box)
        main_vbox.addLayout(tree_box)

        # Log section
        log_label = QLabel("Training Logs:")
        log_label.setFont(QFont("Arial", 9, QFont.Bold))
        main_vbox.addWidget(log_label)
        main_vbox.addWidget(self.log_textedit)

        return main_vbox