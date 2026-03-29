import tkinter as tk
from tkinter import filedialog, messagebox
from model_trainer import ModelTrainer

class TrainPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.dataset_path = None
        self.trainer = ModelTrainer()
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="YOLO Model Training", font=("Arial", 16)).pack(pady=10)

        # Dataset seçimi
        tk.Button(self, text="Select Dataset Folder", command=self.select_dataset).pack(pady=5)
        self.dataset_label = tk.Label(self, text="No dataset selected", fg="gray")
        self.dataset_label.pack()

        # Epochs
        tk.Label(self, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.pack()

        # Batch
        tk.Label(self, text="Batch Size:").pack()
        self.batch_entry = tk.Entry(self)
        self.batch_entry.insert(0, "16")
        self.batch_entry.pack()

        # Img size
        tk.Label(self, text="Image Size:").pack()
        self.imgsize_entry = tk.Entry(self)
        self.imgsize_entry.insert(0, "640")
        self.imgsize_entry.pack()

        # Başlat butonu
        tk.Button(self, text="Start Training", command=self.start_training).pack(pady=10)

    def select_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path = folder
            self.dataset_label.config(text=f"Dataset: {folder}", fg="black")

    def start_training(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return

        try:
            epochs = int(self.epochs_entry.get())
            batch = int(self.batch_entry.get())
            img_size = int(self.imgsize_entry.get())

            messagebox.showinfo("Training", "Training started. Please wait...")

            # ModelTrainer sınıfını kullanarak eğitimi başlat
            self.trainer.start_training(self.dataset_path, epochs, batch, img_size)

            messagebox.showinfo("Success", "Training completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
