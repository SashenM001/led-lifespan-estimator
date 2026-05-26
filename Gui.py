import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import subprocess
from imgp_Project import LEDLifespanEstimator
  # Adjust import if needed

class LEDLifespanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LED Bulb Lifespan Estimator")
        self.estimator = None

        self.frame = tk.Frame(root, padx=20, pady=20)
        self.frame.pack()

        # Directory selection
        tk.Label(self.frame, text="LED Images Folder:").grid(row=0, column=0, sticky="e")
        self.dir_entry = tk.Entry(self.frame, width=50)
        self.dir_entry.grid(row=0, column=1, padx=10)
        tk.Button(self.frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        # Results folder selection (optional)
        tk.Label(self.frame, text="Results Folder:").grid(row=1, column=0, sticky="e")
        self.res_entry = tk.Entry(self.frame, width=50)
        self.res_entry.grid(row=1, column=1, padx=10)
        tk.Button(self.frame, text="Browse", command=self.browse_result_folder).grid(row=1, column=2)

        # Run analysis
        self.run_btn = tk.Button(self.frame, text="Run Analysis", command=self.run_analysis, bg="#4CAF50", fg="white")
        self.run_btn.grid(row=2, column=1, pady=15)

        # Status and summary
        self.status_label = tk.Label(self.frame, text="", fg="blue")
        self.status_label.grid(row=3, column=0, columnspan=3)
        self.summary_text = tk.Text(self.frame, height=6, width=60, state=tk.DISABLED)
        self.summary_text.grid(row=4, column=0, columnspan=3, pady=5)

        # Open plot button (disabled initially)
        self.plot_btn = tk.Button(self.frame, text="Open Results Plot", state=tk.DISABLED, command=self.open_plot)
        self.plot_btn.grid(row=5, column=1)

        self.output_files = {}

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, folder)

    def browse_result_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.res_entry.delete(0, tk.END)
            self.res_entry.insert(0, folder)

    def run_analysis(self):
        images_dir = self.dir_entry.get()
        results_dir = self.res_entry.get() or "results"
        if not images_dir or not os.path.exists(images_dir):
            messagebox.showerror("Error", "Please select a valid LED images folder.")
            return
        self.status_label.config(text="Running analysis, please wait...")
        self.run_btn.config(state=tk.DISABLED)
        self.plot_btn.config(state=tk.DISABLED)
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        threading.Thread(target=self._run_analysis_thread, args=(images_dir, results_dir)).start()

    def _run_analysis_thread(self, images_dir, results_dir):
        try:
            estimator = LEDLifespanEstimator(images_dir, results_dir)
            output = estimator.run_analysis(time_interval_days=30)
            self.output_files = output
            summary = self.make_summary(estimator)
            self.update_status("Analysis complete!", summary)
            self.plot_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.update_status(f"Error: {e}")

        self.run_btn.config(state=tk.NORMAL)

    def update_status(self, msg, summary=None):
        self.status_label.config(text=msg)
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        if summary:
            self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)

    def make_summary(self, estimator):
        if not estimator.history:
            return "No analysis data found."
        latest = estimator.history[-1]
        return (f"Latest Results ({latest['timestamp']}):\n"
                f"Brightness: {latest['norm_brightness']*100:.1f}%\n"
                f"Color Temp Stability: {latest['norm_color_temp']*100:.1f}%\n"
                f"Flicker Index: {latest['flicker_index']*100:.3f}%\n"
                f"Remaining Lifespan: {latest['remaining_lifespan']:.1f}%")

    def open_plot(self):
        plot_path = self.output_files.get('plot')
        if plot_path and os.path.exists(plot_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(plot_path)
                elif os.name == 'posix':  # Mac or Linux
                    subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', plot_path))
            except Exception as e:
                messagebox.showerror("Error", f"Could not open plot: {e}")
        else:
            messagebox.showerror("Error", "Plot file not found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LEDLifespanGUI(root)
    root.mainloop()
