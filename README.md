# LED Bulb Lifespan Estimator

A computer vision-based tool that analyzes LED bulb images over time to estimate their remaining lifespan and predict failure points. The system uses advanced image processing techniques to track brightness degradation, color temperature shifts, and flicker patterns.

## Features

- **Image-Based Analysis**: Processes LED bulb images to extract key degradation indicators
- **Multi-Factor Assessment**: Evaluates brightness, color temperature stability, and flicker patterns
- **Visual Interface**: User-friendly GUI built with Tkinter for easy image batch processing
- **Detailed Processing Steps**: Saves intermediate processing images for transparency and debugging
- **Historical Tracking**: Maintains a comprehensive history of LED health metrics over time
- **Data Export**: Generates CSV reports and matplotlib visualizations

## Technology Stack

- **Python 3.x**
- **OpenCV** - Image processing and feature extraction
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization and plotting
- **Tkinter** - GUI framework
- **Pillow** - Image handling
- **CSV** - Data logging and export

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SashenM001/led-lifespan-estimator.git
cd led-lifespan-estimator
```

2. Install required dependencies:
```bash
pip install opencv-python numpy matplotlib pillow
```

3. (Optional) Install PyInstaller if you want to build an executable:
```bash
pip install pyinstaller
```

## Usage

### Graphical User Interface

Run the GUI application:
```bash
python Gui.py
```

**Steps:**
1. Click "Browse" to select a folder containing LED bulb images
2. (Optional) Select an output folder for results
3. Click "Run Analysis" to start processing
4. View results and click "Open Results Plot" to see visualizations

### Image Requirements
- **Format**: JPG, PNG, or DNG
- **Content**: Clear images of LED bulbs
- **Timing**: Multiple images captured at regular intervals (e.g., every 30 days)

### Output

The analysis generates:
- `processing_steps/` - Intermediate images showing each processing stage
- `LED_health_analysis.csv` - Historical metrics data
- `led_health_report.png` - Comprehensive visualization dashboard
- Summary statistics in the GUI

## Project Structure

```
led-lifespan-estimator/
├── Gui.py                           # Main GUI application
├── imgp_Project.py                  # Core analysis engine
├── README.md                        # This file
└── build/                           # PyInstaller build artifacts
```

## How It Works

### Image Processing Pipeline

1. **Image Loading**: Reads LED bulb images from specified directory
2. **Masking**: Creates adaptive masks to isolate the LED bulb area
3. **Feature Extraction**: Analyzes brightness, color temperature, and flicker
4. **Normalization**: Compares metrics against initial reference values
5. **Health Scoring**: Calculates overall LED health using weighted metrics
6. **Visualization**: Generates plots and reports

### Key Metrics

- **Brightness** (60% weight): Measures luminous intensity degradation
- **Color Temperature Stability** (30% weight): Tracks color shifts over time
- **Flicker** (10% weight): Detects flickering patterns indicating instability

### Lifespan Prediction

The system estimates remaining lifespan by:
- Tracking metric degradation rates
- Comparing against configurable thresholds (default: 70% brightness, 15% color shift)
- Extrapolating failure point based on historical trends

## Configuration

Edit thresholds in `imgp_Project.py`:

```python
self.brightness_threshold = 0.7          # 70% of initial brightness
self.color_temp_threshold = 0.15         # 15% shift in color temperature
self.flicker_threshold = 0.1             # 10% increase in flicker

self.weights = {
    'brightness': 0.6,
    'color_temp': 0.3,
    'flicker': 0.1
}
```

## Building Executable

To create a standalone Windows/Mac executable:

```bash
pyinstaller Gui.spec
```

The executable will be in the `build/` directory.

## Results Interpretation

- **Brightness**: Percentage of initial luminous intensity
- **Color Temperature Stability**: Deviation from baseline color
- **Health Score**: Composite metric (0-100) indicating overall LED condition
- **Estimated Lifespan**: Projected hours/days remaining

## Limitations

- Requires consistent lighting conditions during image capture
- LED bulb must remain in the same position
- Image quality affects accuracy
- Initial reference image is critical for baseline metrics

## Future Enhancements

- [ ] Real-time LED monitoring with webcam
- [ ] Machine learning-based failure prediction
- [ ] Multi-bulb simultaneous monitoring
- [ ] Cloud-based data storage
- [ ] Mobile app integration

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source. See LICENSE file for details.

## Author

**Sashend M.**

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Note**: This tool is designed for research and educational purposes. Actual LED lifespan varies based on manufacturer specifications, environmental conditions, and usage patterns.
