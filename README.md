# 🪖 Helmet Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A real-time helmet detection system using OpenCV and Python to enhance workplace and road safety.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 📌 Overview

The **Helmet Detection System** is a computer vision project that automatically detects whether individuals in images or video streams are wearing helmets. Built using **Python** and **OpenCV**, it is designed for safety monitoring in construction sites, factories, highways, and two-wheeler traffic enforcement.

> ⚡ Real-time detection | 🎯 High accuracy | 🔧 Easy to configure

---

## ✨ Features

- ✅ **Real-time detection** from webcam or video file
- ✅ **Image-based detection** for static photos
- ✅ **Bounding box visualization** with confidence scores
- ✅ **Alert system** — triggers warning when helmet is not detected
- ✅ **Support for multiple persons** in a single frame
- ✅ **Lightweight and fast** — runs on standard hardware
- ✅ **Configurable thresholds** for detection sensitivity
- ✅ **Log output** — saves detection results to a CSV/log file

---

## 🎬 Demo

```
Input Frame  →  Person Detected  →  Helmet Check  →  Output (With/Without Helmet Label)
```

| With Helmet | Without Helmet |
|---|---|
| ✅ Green bounding box | ❌ Red bounding box + Alert |

> 📸 *(Add your demo screenshots or GIF here)*
> 
> Example: `![Demo](assets/demo.gif)`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| OpenCV (`cv2`) | Image/video processing, frame capture |
| NumPy | Array manipulation and image math |
| Haar Cascade / YOLOv5 | Object detection model |
| Matplotlib | Visualization and result plotting |
| imutils | Convenience utilities for OpenCV |

---

## ⚙️ Installation

### Prerequisites

Make sure you have Python 3.8 or above installed.

```bash
python --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/Avii-Kumar18/helmet-detection-system.git
cd helmet-detection-system
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
imutils>=0.5.4
```

---

## 🚀 Usage

### Detect from Webcam (Real-time)

```bash
python detect.py --source webcam
```

### Detect from a Video File

```bash
python detect.py --source video --input path/to/video.mp4
```

### Detect from an Image

```bash
python detect.py --source image --input path/to/image.jpg
```

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `webcam` | Input source: `webcam`, `video`, `image` |
| `--input` | `None` | Path to video or image file |
| `--confidence` | `0.5` | Minimum confidence threshold (0.0 – 1.0) |
| `--output` | `output/` | Folder to save result frames |
| `--save-log` | `False` | Save detection log to CSV |
| `--display` | `True` | Show live detection window |

**Example with all arguments:**

```bash
python detect.py --source video --input traffic.mp4 --confidence 0.6 --output results/ --save-log True
```

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                        │
│                                                              │
│  [Input]  →  [Preprocessing]  →  [Person Detection]         │
│                                         ↓                    │
│  [Output] ←  [Labeling & Alert] ← [Helmet Classification]   │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-step breakdown

1. **Frame Capture** — OpenCV reads frames from webcam, video file, or image.
2. **Preprocessing** — Frame is resized, converted to grayscale/blob as needed.
3. **Person / Head Detection** — A Haar Cascade or deep learning model detects human heads/upper bodies.
4. **ROI Extraction** — Region of Interest (ROI) is cropped around each detected head.
5. **Helmet Classification** — The ROI is passed through a classifier to determine: `Helmet` or `No Helmet`.
6. **Annotation** — Bounding boxes are drawn:
   - 🟢 **Green** → Helmet detected
   - 🔴 **Red** → No helmet detected
7. **Alert / Logging** — If no helmet is found, an alert message is displayed and optionally logged.

---

## 📁 Project Structure

```
helmet-detection-system/
│
├── assets/                  # Demo images, GIFs, screenshots
│   └── demo.gif
│
├── models/                  # Pre-trained model weights
│   ├── haarcascade_head.xml
│   └── helmet_classifier.h5
│
├── data/                    # Sample images and test videos
│   ├── images/
│   └── videos/
│
├── output/                  # Detection results saved here
│
├── detect.py                # Main detection script
├── classifier.py            # Helmet classification logic
├── utils.py                 # Helper functions (drawing, logging)
├── config.py                # Configuration and constants
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

---

## 🔍 Model Details

### Option A — Haar Cascade (Lightweight)
- Uses OpenCV's built-in `CascadeClassifier`
- Fast and runs on CPU
- Best for simple use cases and low-end hardware

### Option B — Deep Learning (YOLOv5 / Custom CNN)
- Higher accuracy, handles complex backgrounds
- Requires more compute resources
- Recommended for production use

You can switch between models in `config.py`:

```python
# config.py
MODEL_TYPE = "haar"        # Options: "haar" | "yolo" | "cnn"
CONFIDENCE_THRESHOLD = 0.5
HELMET_LABEL = "Helmet"
NO_HELMET_LABEL = "No Helmet"
```

---

## 📊 Sample Output

```
[INFO] Starting detection on: traffic.mp4
[INFO] Frame 1/240 — 2 persons detected
  → Person 1: Helmet ✅ (confidence: 0.91)
  → Person 2: No Helmet ❌ (confidence: 0.87) — ALERT!
[INFO] Detection complete. Results saved to output/results.csv
```

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** your feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "Add: your feature description"`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

---

## 🐛 Known Issues / Limitations

- Detection accuracy may drop in low-light conditions
- Helmet detection may be less accurate with non-standard helmets (e.g., bicycle helmets)
- Performance may vary with high-resolution video on low-end hardware

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Avinash Kumar**  
AI/ML Engineer | B.Tech CSE (AI), CSVTU Bhilai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/avinash-kumar-130653404/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Avii-Kumar18)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:heyyavi2005@gmail.com)

---

## ⭐ Show Your Support

If you found this project useful, please consider giving it a **⭐ star** on GitHub — it really helps!

---

<div align="center">
  <sub>Built with ❤️ using Python & OpenCV</sub>
</div>
