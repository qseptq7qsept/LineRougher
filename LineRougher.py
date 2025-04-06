import sys
import os
import json
import cv2
import numpy as np
import random
import subprocess

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QSlider, QFileDialog, QGraphicsScene, QGraphicsView, QCheckBox, QInputDialog
)
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor
from PySide6.QtCore import Qt, QThread, Signal, QTimer

def compute_line_alpha_soft_threshold(gray, black_threshold, tolerance):
    """
    Returns an alpha channel [0..255] where:
      if g <= black_threshold: alpha=255 (fully black)
      if g >= black_threshold + tolerance: alpha=0 (not black at all)
      else alpha = linear ramp from 255..0 over that range
    """
    gray_f = gray.astype(np.float32)
    alpha_f = np.zeros_like(gray_f, dtype=np.float32)

    lower = float(black_threshold)
    upper = float(black_threshold + tolerance)

    alpha_f[gray_f <= lower] = 1.0
    in_ramp = (gray_f > lower) & (gray_f < upper)
    alpha_f[in_ramp] = 1.0 - (gray_f[in_ramp] - lower) / (upper - lower)

    alpha = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)
    return alpha

def composite_alpha_to_white(img):
    """
    If the image has an alpha channel, composite onto a white background.
    Returns a BGR image with no alpha channel.
    """
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
        img_bgr = (white_bg * (1 - alpha[..., None]) + img[:, :, :3] * alpha[..., None]).astype(np.uint8)
        return img_bgr
    else:
        return img

def to_black_alpha(scribbled_bgr):
    """
    Converts the scribbled BGR image into an image with only:
      - Black (fully opaque),
      - Semi-transparent black,
      - Fully transparent.
    We do this by converting to grayscale and setting:
        alpha = 255 - gray_value
    The RGB channels are set to 0 (black).
    """
    h, w, _ = scribbled_bgr.shape
    gray = cv2.cvtColor(scribbled_bgr, cv2.COLOR_BGR2GRAY)
    alpha = 255 - gray
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[..., 3] = alpha
    return result

def multiply_blend(source, processed):
    """
    Multiply blend two BGR images: out = (source * processed)/255.
    """
    src_f = source.astype(np.float32)
    proc_f = processed.astype(np.float32)
    blended = (src_f * proc_f) / 255.0
    return blended.astype(np.uint8)

class ImageProcessor:
    """
    Transforms a clean line-art image into a rough, scribbly version.
    Also duplicates some lines (per segment) with slight modifications.
    Uses threshold + tolerance to detect black regions.
    """
    def __init__(self):
        self.black_threshold = 72  
        self.tolerance = 20        
        self.step_size = 11
        self.noise_amount = 0
        self.line_thickness = 1
        self.scribble_passes = 2
        self.join_distance = 0
        self.duplication_chance = 1.0
        self.duplicate_scale = 1.0

    def process_image(self, image):
        mask = self.binarize_line_art(image)
        scribbled = self.draw_rough_contours(
            mask,
            step=self.step_size,
            noise=self.noise_amount,
            thickness=self.line_thickness,
            passes=self.scribble_passes,
            color=(0, 0, 0)
        )
        if self.join_distance > 0:
            scribbled = self.apply_join_distance(scribbled, self.join_distance)
        return scribbled

    def binarize_line_art(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        alpha_soft = compute_line_alpha_soft_threshold(gray, self.black_threshold, self.tolerance)
        mask = np.where(alpha_soft >= 128, 255, 0).astype(np.uint8)
        return mask

    def apply_join_distance(self, scribbled, distance):
        gray = cv2.cvtColor(scribbled, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((distance, distance), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        final = np.ones_like(scribbled, dtype=np.uint8) * 255
        final[closed == 255] = [0, 0, 0]
        return final

    def draw_rough_contours(self, mask, step=2, noise=3, thickness=2, passes=1, color=(0, 0, 0)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        seg_length = 15

        for _ in range(passes):
            for contour in contours:
                if len(contour) < 2:
                    continue
                sampled_pts = self.resample_contour(contour, step=step)
                rough_pts = self.roughen_contour(sampled_pts, noise=noise)
                for i in range(len(rough_pts) - 1):
                    cv2.line(canvas, rough_pts[i], rough_pts[i+1], color, thickness, cv2.LINE_AA)
                n = len(rough_pts)
                start = 0
                while start < n - 1:
                    end = min(start + seg_length, n)
                    segment = rough_pts[start:end]
                    if len(segment) >= 2 and random.random() < self.duplication_chance:
                        dup_segment = self.duplicate_contour(segment)
                        for i in range(len(dup_segment) - 1):
                            cv2.line(canvas, dup_segment[i], dup_segment[i+1], color, thickness, cv2.LINE_AA)
                    start += seg_length
        return canvas

    def resample_contour(self, contour, step=2):
        pts = contour[:, 0, :]
        distances = [0.0]
        for i in range(1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[i-1])
            distances.append(distances[-1] + dist)
        total_length = distances[-1]
        if total_length == 0:
            return pts.tolist()
        num_samples = int(total_length // step)
        if num_samples < 2:
            return pts.tolist()
        resampled = []
        current_index = 0
        for s in range(num_samples + 1):
            d = s * step
            while current_index < len(distances) - 1 and distances[current_index+1] < d:
                current_index += 1
            if current_index == len(distances) - 1:
                resampled.append(pts[-1])
            else:
                seg_len = distances[current_index+1] - distances[current_index]
                ratio = 0 if seg_len == 0 else (d - distances[current_index]) / seg_len
                p1 = pts[current_index]
                p2 = pts[current_index+1]
                px = p1[0] + (p2[0] - p1[0]) * ratio
                py = p1[1] + (p2[1] - p1[1]) * ratio
                resampled.append((px, py))
        return resampled

    def roughen_contour(self, points, noise=3):
        rough_pts = []
        for (x, y) in points:
            nx = x + random.randint(-noise, noise)
            ny = y + random.randint(-noise, noise)
            rough_pts.append((int(nx), int(ny)))
        return rough_pts

    def duplicate_contour(self, points):
        pts_np = np.array(points, dtype=np.float32)
        centroid = np.mean(pts_np, axis=0)
        random_angle = random.uniform(-5, 5)
        random_scale = random.uniform(0.95, 1.05)
        final_scale = random_scale * self.duplicate_scale
        rad = np.deg2rad(random_angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        dup_pts = []
        for pt in pts_np:
            offset = pt - centroid
            offset *= final_scale
            x_new = offset[0] * cos_a - offset[1] * sin_a
            y_new = offset[0] * sin_a + offset[1] * cos_a
            new_pt = centroid + [x_new, y_new]
            dup_pts.append((int(new_pt[0]), int(new_pt[1])))
        return np.array(dup_pts, dtype=np.int32).tolist()

class ProcessWorker(QThread):
    finished = Signal(np.ndarray)
    def __init__(self, image, processor):
        super().__init__()
        self.image = image
        self.processor = processor
    def run(self):
        result = self.processor.process_image(self.image)
        self.finished.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Rougher v1.2 -q7")
        self.processor = ImageProcessor()
        self.image = None         # Source image (BGR)
        self.processed_image = None  # Scribbled image (BGR)
        self.worker = None
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(200)
        self.preview_timer.timeout.connect(self.run_processing)
        self.setup_ui()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Graphics view for live preview.
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.main_layout.addWidget(self.graphics_view)

        # Checkbox for viewer overlay.
        self.checkbox_viewer_overlay = QCheckBox("Show Rough on Source")
        self.checkbox_viewer_overlay.stateChanged.connect(self.refresh_preview)
        self.main_layout.addWidget(self.checkbox_viewer_overlay)

        def add_control(label_text, min_val, max_val, initial, callback):
            layout = QHBoxLayout()
            label = QLabel(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(initial)
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(initial)
            slider.valueChanged.connect(lambda v: spin.setValue(v))
            spin.valueChanged.connect(lambda v: slider.setValue(v))
            spin.valueChanged.connect(callback)
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.addWidget(spin)
            return layout

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(add_control(
            "Black Threshold:", 0, 255, self.processor.black_threshold,
            lambda v: self.update_param("black_threshold", v)
        ))
        controls_layout.addLayout(add_control(
            "Tolerance:", 0, 100, self.processor.tolerance,
            lambda v: self.update_param("tolerance", v)
        ))
        controls_layout.addLayout(add_control(
            "Step:", 1, 20, self.processor.step_size,
            lambda v: self.update_param("step_size", v)
        ))
        controls_layout.addLayout(add_control(
            "Noise:", 0, 30, self.processor.noise_amount,
            lambda v: self.update_param("noise_amount", v)
        ))
        controls_layout.addLayout(add_control(
            "Thickness:", 1, 10, self.processor.line_thickness,
            lambda v: self.update_param("line_thickness", v)
        ))
        controls_layout.addLayout(add_control(
            "Passes:", 1, 10, self.processor.scribble_passes,
            lambda v: self.update_param("scribble_passes", v)
        ))
        controls_layout.addLayout(add_control(
            "Join Distance:", 0, 20, self.processor.join_distance,
            lambda v: self.update_param("join_distance", v)
        ))
        dup_chance_layout = QHBoxLayout()
        dup_chance_label = QLabel("Duplication Chance (%):")
        dup_chance_slider = QSlider(Qt.Horizontal)
        dup_chance_slider.setRange(0, 100)
        dup_chance_slider.setValue(int(self.processor.duplication_chance * 100))
        dup_chance_spin = QSpinBox()
        dup_chance_spin.setRange(0, 100)
        dup_chance_spin.setValue(int(self.processor.duplication_chance * 100))
        dup_chance_slider.valueChanged.connect(lambda v: dup_chance_spin.setValue(v))
        dup_chance_spin.valueChanged.connect(lambda v: dup_chance_slider.setValue(v))
        dup_chance_spin.valueChanged.connect(lambda v: self.update_param("duplication_chance", v / 100.0))
        dup_chance_layout.addWidget(dup_chance_label)
        dup_chance_layout.addWidget(dup_chance_slider)
        dup_chance_layout.addWidget(dup_chance_spin)
        controls_layout.addLayout(dup_chance_layout)
        dup_scale_layout = QHBoxLayout()
        dup_scale_label = QLabel("Duplicate Scale (%):")
        dup_scale_slider = QSlider(Qt.Horizontal)
        dup_scale_slider.setRange(50, 200)
        dup_scale_slider.setValue(int(self.processor.duplicate_scale * 100))
        dup_scale_spin = QSpinBox()
        dup_scale_spin.setRange(50, 200)
        dup_scale_spin.setValue(int(self.processor.duplicate_scale * 100))
        dup_scale_slider.valueChanged.connect(lambda v: dup_scale_spin.setValue(v))
        dup_scale_spin.valueChanged.connect(lambda v: dup_scale_slider.setValue(v))
        dup_scale_spin.valueChanged.connect(lambda v: self.update_param("duplicate_scale", v / 100.0))
        dup_scale_layout.addWidget(dup_scale_label)
        dup_scale_layout.addWidget(dup_scale_slider)
        dup_scale_layout.addWidget(dup_scale_spin)
        controls_layout.addLayout(dup_scale_layout)

        self.main_layout.addLayout(controls_layout)

        # Checkbox for saving overlay.
        self.checkbox_save_overlay = QCheckBox("Save Rough on Source (No Alpha)")
        self.main_layout.addWidget(self.checkbox_save_overlay)

        # Buttons layout.
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.batch_button = QPushButton("Batch Process Folder")
        self.batch_button.clicked.connect(self.batch_process)
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.load_config_button_clicked)
        # New button for video-to-image sequence conversion.
        self.vid2seq_button = QPushButton("Video to Image Seq")
        self.vid2seq_button.clicked.connect(self.vid2seq)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.batch_button)
        button_layout.addWidget(self.save_config_button)
        button_layout.addWidget(self.load_config_button)
        button_layout.addWidget(self.vid2seq_button)
        self.main_layout.addLayout(button_layout)

    def refresh_preview(self):
        if self.processed_image is not None:
            self.on_processing_finished(self.processed_image)

    def update_param(self, name, value):
        setattr(self.processor, name, value)
        self.preview_timer.start()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            if img is None:
                print("Error loading image!")
                return
            img = composite_alpha_to_white(img)
            self.image = img
            self.preview_timer.start()

    def save_image(self):
        if self.processed_image is None or self.image is None:
            print("No processed image to save.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if file_name:
            if self.checkbox_save_overlay.isChecked():
                overlay = multiply_blend(self.image, self.processed_image)
                cv2.imwrite(file_name, overlay)
                print(f"Saved overlaid image to {file_name}")
            else:
                black_alpha = to_black_alpha(self.processed_image)
                cv2.imwrite(file_name, black_alpha)
                print(f"Saved scribbled image (black+alpha) to {file_name}")

    def batch_process(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            output_folder = os.path.join(folder, "processed_scribbles")
            os.makedirs(output_folder, exist_ok=True)
            for file in os.listdir(folder):
                lower = file.lower()
                if lower.endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(folder, file)
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    img = composite_alpha_to_white(img)
                    processed = self.processor.process_image(img)
                    if self.checkbox_save_overlay.isChecked():
                        result = multiply_blend(img, processed)
                    else:
                        result = to_black_alpha(processed)
                    out_name = os.path.splitext(file)[0] + ".png"
                    output_path = os.path.join(output_folder, out_name)
                    cv2.imwrite(output_path, result)
            print(f"Batch processing completed. Output in '{output_folder}'.")

    def save_config(self):
        config = {
            "black_threshold": self.processor.black_threshold,
            "tolerance": self.processor.tolerance,
            "step_size": self.processor.step_size,
            "noise_amount": self.processor.noise_amount,
            "line_thickness": self.processor.line_thickness,
            "scribble_passes": self.processor.scribble_passes,
            "join_distance": self.processor.join_distance,
            "duplication_chance": self.processor.duplication_chance,
            "duplicate_scale": self.processor.duplicate_scale
        }
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Config", "config.json", "JSON Files (*.json)")
        if file_name:
            with open(file_name, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Config saved to {file_name}")

    def load_config_button_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if file_name:
            self.load_config(file_name)

    def load_config(self, file_name):
        try:
            with open(file_name, "r") as f:
                data = json.load(f)
            self.apply_config(data)
            print(f"Config loaded from {file_name}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def apply_config(self, data):
        self.processor.black_threshold = data.get("black_threshold", self.processor.black_threshold)
        self.processor.tolerance = data.get("tolerance", self.processor.tolerance)
        self.processor.step_size = data.get("step_size", self.processor.step_size)
        self.processor.noise_amount = data.get("noise_amount", self.processor.noise_amount)
        self.processor.line_thickness = data.get("line_thickness", self.processor.line_thickness)
        self.processor.scribble_passes = data.get("scribble_passes", self.processor.scribble_passes)
        self.processor.join_distance = data.get("join_distance", self.processor.join_distance)
        self.processor.duplication_chance = data.get("duplication_chance", self.processor.duplication_chance)
        self.processor.duplicate_scale = data.get("duplicate_scale", self.processor.duplicate_scale)
        self.reinitialize_ui()
        self.preview_timer.start()

    def reinitialize_ui(self):
        self.main_layout.removeWidget(self.graphics_view)
        self.graphics_view.deleteLater()
        self.setup_ui()
        self.main_layout.insertWidget(0, self.graphics_view)
        self.graphics_view.setScene(self.scene)
        self.update()

    def run_processing(self):
        if self.image is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        self.worker = ProcessWorker(self.image, self.processor)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, result):
        self.processed_image = result
        if self.checkbox_viewer_overlay.isChecked() and self.image is not None:
            display_img = multiply_blend(self.image, self.processed_image)
        else:
            display_img = self.processed_image
        image_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene.items():
            self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def vid2seq(self):
        # Open file dialog to select a video file.
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_name:
            return
        # Popup to set the sequence name.
        sequence_name, ok = QInputDialog.getText(self, "Sequence Name", "Enter sequence name:", text="UserInput")
        if not ok or not sequence_name:
            return
        source_folder = os.path.dirname(file_name)

        # Use the script's directory to build the ffmpeg path.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_path = os.path.join(script_dir, "ffmpeg", "ffmpeg.exe")

        # Create a subfolder for the output images.
        output_folder = os.path.join(source_folder, sequence_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Build the output pattern.
        output_pattern = os.path.join(output_folder, f"{sequence_name}_%04d.png")
        
        # Build and run the ffmpeg command.
        command = f'"{ffmpeg_path}" -i "{file_name}" "{output_pattern}"'
        subprocess.run(command, shell=True)
        print(f"Video conversion completed. Images saved as {sequence_name}_XXXX.png in {output_folder}.")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53,53,53))
    dark_palette.setColor(QPalette.WindowText, QColor(255,255,255))
    dark_palette.setColor(QPalette.Base, QColor(42,42,42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66,66,66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255,255,255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255,255,255))
    dark_palette.setColor(QPalette.Text, QColor(255,255,255))
    dark_palette.setColor(QPalette.Button, QColor(53,53,53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255,255,255))
    dark_palette.setColor(QPalette.BrightText, QColor(255,0,0))
    dark_palette.setColor(QPalette.Link, QColor(208,42,218))
    dark_palette.setColor(QPalette.Highlight, QColor(208,42,218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0,0,0))
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
