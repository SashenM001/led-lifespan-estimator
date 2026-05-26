import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

class LEDLifespanEstimator:
    def __init__(self, images_dir, output_dir="results"):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.reference_brightness = None
        self.reference_color_temp = None
        self.history = []
        self.timestamps = []
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.steps_dir = os.path.join(output_dir, "processing_steps")
        if not os.path.exists(self.steps_dir): os.makedirs(self.steps_dir)
        self.brightness_threshold = 0.7
        self.color_temp_threshold = 0.15
        self.flicker_threshold = 0.1
        self.weights = {'brightness': 0.6, 'color_temp': 0.3, 'flicker': 0.1}

    def load_images(self):
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png', 'DNG'))]
        image_files.sort()
        return [os.path.join(self.images_dir, f) for f in image_files]

    def save_step_image(self, image, step_name, image_index, original_filename):
        base_name = os.path.splitext(original_filename)[0]
        step_filename = f"{base_name}_{image_index:02d}_{step_name}.png"
        step_path = os.path.join(self.steps_dir, step_filename)
        cv2.imwrite(step_path, image)

    def create_visualization_image(self, original, masked, hsv, lab, gray, mask, image_index, original_filename):
        h, w = original.shape[:2]
        composite = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        steps = [
            (original, "1. Original"), (masked, "2. Masked"),
            (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "3. HSV"),
            (cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "4. LAB"),
            (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "5. Grayscale"),
            (cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), "6. Adaptive Mask")
        ]
        for i, (img, label) in enumerate(steps):
            row, col = i // 3, i % 3
            composite[row*h:(row+1)*h, col*w:(col+1)*w] = img
            cv2.putText(composite, label, (col*w + 10, row*h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        base_name = os.path.splitext(original_filename)[0]
        composite_filename = f"{base_name}_{image_index:02d}_all_steps.png"
        composite_path = os.path.join(self.steps_dir, composite_filename)
        cv2.imwrite(composite_path, composite)

    def create_adaptive_mask(self, img, image_index, original_filename):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.save_step_image(gray, "02_grayscale_initial", image_index, original_filename)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        self.save_step_image(blurred, "03_blurred", image_index, original_filename)
        bright_threshold = np.max(gray) * 0.6
        _, bright_mask = cv2.threshold(blurred, bright_threshold, 255, cv2.THRESH_BINARY)
        self.save_step_image(bright_mask, "04_bright_threshold", image_index, original_filename)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_step_image(otsu_mask, "05_otsu_threshold", image_index, original_filename)
        combined_mask = cv2.bitwise_or(bright_mask, otsu_mask)
        self.save_step_image(combined_mask, "06_combined_mask", image_index, original_filename)
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        self.save_step_image(cleaned_mask, "07_opened_mask", image_index, original_filename)
        filled_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        self.save_step_image(filled_mask, "08_filled_mask", image_index, original_filename)
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        self.save_step_image(contour_img, "09_contours", image_index, original_filename)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(final_mask, [largest_contour], 255)
            area = cv2.contourArea(largest_contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    hull = cv2.convexHull(largest_contour)
                    cv2.fillPoly(final_mask, [hull], 255)
        else:
            final_mask = filled_mask
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        self.save_step_image(final_mask, "10_final_mask", image_index, original_filename)
        mask_overlay = img.copy()
        green_overlay = np.zeros_like(img)
        green_overlay[final_mask > 0] = [0,255,0]
        mask_overlay = cv2.addWeighted(mask_overlay, 0.7, green_overlay, 0.3, 0)
        self.save_step_image(mask_overlay, "11_mask_overlay", image_index, original_filename)
        return final_mask > 0

    def extract_features(self, image_path, image_index=0):
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"Could not read image at {image_path}")
        original_filename = os.path.basename(image_path)
        self.save_step_image(img, "01_original", image_index, original_filename)
        mask = self.create_adaptive_mask(img, image_index, original_filename)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        mask_img = (mask * 255).astype(np.uint8)
        self.save_step_image(mask_img, "12_adaptive_mask", image_index, original_filename)
        img_masked = img.copy(); img_masked[~mask_3ch] = 0
        self.save_step_image(img_masked, "13_masked", image_index, original_filename)
        hsv_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2LAB)
        gray_img = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        self.save_step_image(hsv_img, "14_hsv", image_index, original_filename)
        self.save_step_image(lab_img, "15_lab", image_index, original_filename)
        self.save_step_image(gray_img, "16_grayscale", image_index, original_filename)
        brightness = np.mean(hsv_img[:, :, 2][mask]) / 255.0
        brightness_img = hsv_img[:, :, 2].copy(); brightness_img[~mask] = 0
        self.save_step_image(brightness_img, "17_brightness_channel", image_index, original_filename)
        a_channel = np.mean(lab_img[:, :, 1][mask])
        b_channel = np.mean(lab_img[:, :, 2][mask])
        color_temp = np.sqrt(a_channel**2 + b_channel**2)
        a_img = lab_img[:, :, 1].copy(); b_img = lab_img[:, :, 2].copy()
        a_img[~mask] = 128; b_img[~mask] = 128
        self.save_step_image(a_img, "18_a_channel", image_index, original_filename)
        self.save_step_image(b_img, "19_b_channel", image_index, original_filename)
        if self.reference_brightness is None:
            self.reference_brightness = brightness
            self.reference_color_temp = color_temp
            flicker_index = 0.0; diff_img = np.zeros_like(gray_img)
        else:
            if hasattr(self, 'prev_gray'):
                diff_img = cv2.absdiff(gray_img, self.prev_gray)
                flicker_index = np.mean(diff_img[mask]) / 255.0
                self.save_step_image(diff_img, "20_flicker_diff", image_index, original_filename)
            else:
                flicker_index = 0.0; diff_img = np.zeros_like(gray_img)
        self.prev_gray = gray_img.copy()
        norm_brightness = brightness / self.reference_brightness if self.reference_brightness > 0 else 0
        norm_color_temp = 1.0 - abs(color_temp - self.reference_color_temp) / self.reference_color_temp if self.reference_color_temp > 0 else 0
        result_img = img_masked.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; color = (0, 255, 0); thickness = 2
        text_lines = [
            f"Brightness: {brightness:.3f} ({norm_brightness:.3f})",
            f"Color Temp: {color_temp:.2f} ({norm_color_temp:.3f})",
            f"Flicker: {flicker_index:.5f}",
            f"Mask Area: {np.sum(mask)} pixels"
        ]
        y_offset = 30
        for i, line in enumerate(text_lines):
            cv2.putText(result_img, line, (10, y_offset + i * 25), font, font_scale, color, thickness)
        self.save_step_image(result_img, "21_final_result", image_index, original_filename)
        self.create_visualization_image(img, img_masked, hsv_img, lab_img, gray_img, mask, image_index, original_filename)
        return {
            'brightness': brightness,
            'color_temp': color_temp,
            'flicker_index': flicker_index,
            'norm_brightness': norm_brightness,
            'norm_color_temp': norm_color_temp,
            'raw_img': img,
            'gray_img': gray_img,
            'diff_img': diff_img,
            'mask': mask
        }

    def estimate_lifespan(self, features):
        brightness_score = min(1.0, features['norm_brightness'] / self.brightness_threshold)
        color_temp_score = features['norm_color_temp']
        flicker_ratio = features['flicker_index'] / self.flicker_threshold
        flicker_score = max(0.0, 1.0 - flicker_ratio)
        health_score = (self.weights['brightness'] * brightness_score +
                        self.weights['color_temp'] * color_temp_score +
                        self.weights['flicker'] * flicker_score)
        return min(max(health_score * 100, 0), 100)

    def analyze_images(self, time_interval_days=1):
        image_paths = self.load_images()
        if not image_paths: return []
        for i, image_path in enumerate(image_paths):
            timestamp = datetime.fromtimestamp(os.path.getmtime(image_path))
            features = self.extract_features(image_path, i)
            remaining_lifespan = self.estimate_lifespan(features)
            entry = {
                'timestamp': timestamp,
                'image_path': image_path,
                'brightness': features['brightness'],
                'norm_brightness': features['norm_brightness'],
                'color_temp': features['color_temp'],
                'norm_color_temp': features['norm_color_temp'],
                'flicker_index': features['flicker_index'],
                'remaining_lifespan': remaining_lifespan
            }
            self.history.append(entry)
            self.timestamps.append(timestamp)
        return self.history

    def plot_results(self):
        if not self.history: return None
        x_labels = [os.path.basename(entry['image_path']) for entry in self.history]
        x_positions = list(range(len(x_labels)))
        brightness_values = [entry['norm_brightness'] * 100 for entry in self.history]
        color_temp_values = [entry['norm_color_temp'] * 100 for entry in self.history]
        flicker_values = [entry['flicker_index'] * 100 for entry in self.history]
        lifespan_values = [entry['remaining_lifespan'] for entry in self.history]
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_positions, brightness_values, 'b-o', linewidth=2)
        ax1.axhline(y=self.brightness_threshold * 100, color='r', linestyle='--', label=f'Threshold ({self.brightness_threshold * 100:.0f}%)')
        ax1.set_title('Normalized Brightness Over Time')
        ax1.set_ylabel('Brightness (%)')
        ax1.set_ylim(0, 110)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.grid(True)
        ax1.legend()
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x_positions, color_temp_values, 'g-o', linewidth=2)
        ax2.set_title('Normalized Color Temperature Over Time')
        ax2.set_ylabel('Color Temp Stability (%)')
        ax2.set_ylim(0, 110)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.grid(True)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(x_positions, flicker_values, 'm-o', linewidth=2)
        ax3.set_title('Flicker Index Over Time')
        ax3.set_ylabel('Flicker Index (%)')
        ax3.set_ylim(0, max(flicker_values) * 1.2 if flicker_values else 10)
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(x_labels, rotation=45)
        ax3.grid(True)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(x_positions, lifespan_values, 'r-o', linewidth=2)
        ax4.set_title('Estimated Remaining Lifespan Over Time')
        ax4.set_ylabel('Remaining Lifespan (%)')
        ax4.set_ylim(0, 110)
        ax4.set_xticks(x_positions)
        ax4.set_xticklabels(x_labels, rotation=45)
        ax4.grid(True)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(x_positions, brightness_values, 'b-o', linewidth=2, label='Brightness')
        ax5.plot(x_positions, color_temp_values, 'g-o', linewidth=2, label='Color Temp Stability')
        ax5.plot(x_positions, flicker_values, 'm-o', linewidth=2, label='Flicker Index')
        ax5.plot(x_positions, lifespan_values, 'r-o', linewidth=2, label='Remaining Lifespan')
        ax5.set_title('LED Bulb Health Parameters Over Time')
        ax5.set_ylabel('Value (%)')
        ax5.set_ylim(0, 110)
        ax5.set_xticks(x_positions)
        ax5.set_xticklabels(x_labels, rotation=45)
        ax5.grid(True)
        ax5.legend()
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'led_lifespan_trends.png')
        plt.savefig(fig_path, dpi=300)
        return fig

    def export_csv(self):
        if not self.history: return None
        csv_path = os.path.join(self.output_dir, 'led_lifespan_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'image_path', 'brightness', 'norm_brightness', 'color_temp', 'norm_color_temp', 'flicker_index', 'remaining_lifespan']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.history:
                entry_copy = entry.copy()
                entry_copy['timestamp'] = entry_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow(entry_copy)
        return csv_path

    def run_analysis(self, time_interval_days=1):
        self.analyze_images(time_interval_days)
        self.plot_results()
        self.export_csv()

def simulate_led_degradation(output_dir="sample_images", num_images=10, width=400, height=400, base_brightness=200, brightness_decay_rate=0.05, color_temp_shift_rate=0.02, flicker_increase_rate=0.01):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    image_paths = []
    base_blue, base_green, base_red = 180, 200, 220
    Y, X = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    radius = min(width, height) // 3
    dist_from_center = ((Y - center[0]) ** 2 + (X - center[1]) ** 2)
    mask = dist_from_center <= radius ** 2
    np.random.seed(42)
    for i in range(num_images):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        current_brightness = base_brightness * (1.0 - brightness_decay_rate * i)
        current_blue = min(255, base_blue * (1.0 + color_temp_shift_rate * i))
        current_green = base_green * (1.0 - color_temp_shift_rate * 0.5 * i)
        current_red = base_red * (1.0 - color_temp_shift_rate * i)
        bulb_color = (int(current_blue), int(current_green), int(current_red))
        img[mask] = bulb_color
        flicker_noise = np.random.normal(0, flicker_increase_rate * (i + 1) * 15, (height, width, 3)).astype(np.int16)
        temp_img = img.copy().astype(np.int16)
        temp_img[mask] += flicker_noise[mask]
        temp_img = np.clip(temp_img, 0, 255).astype(np.uint8)
        center_point = (width // 2, height // 2)
        cv2.circle(temp_img, center_point, radius, (100, 100, 100), 2)
        blurred = cv2.GaussianBlur(temp_img, (21, 21), 0)
        img = cv2.addWeighted(temp_img, 0.7, blurred, 0.3, 0)
        image_path = os.path.join(output_dir, f"led_bulb_{i+1:02d}.png")
        cv2.imwrite(image_path, img)
        image_paths.append(image_path)
    return image_paths

def main():
    sample_images_dir = "C:\\Users\\sashen\\Desktop\\image P\\source"
    results_dir = "C:\\Users\\sashen\\Desktop\\image P\\Results"
    if not os.path.exists(sample_images_dir) or not os.listdir(sample_images_dir):
        simulate_led_degradation(sample_images_dir, num_images=10)
    estimator = LEDLifespanEstimator(sample_images_dir, results_dir)
    estimator.run_analysis(time_interval_days=30)
    plt.show()

if __name__ == "__main__":
    main()
