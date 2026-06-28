# Object-Detection-using-YOLOv8
Real-time object detection using YOLOv8 with high-speed inference on images, videos, and live webcam streams.
# 🚀 YOLOv8 Real-Time Object Detection

A real-time object detection project built using **Ultralytics YOLOv8** and **OpenCV**. This application detects multiple objects from images, videos, and live webcam feeds with customizable confidence thresholds, color-coded bounding boxes, and live FPS monitoring.

---

## 📌 Features

- 🎯 Real-time object detection using YOLOv8
- 📷 Supports:
  - Images
  - Videos
  - Live Webcam
- 📦 Multiple object detection
- 🖍️ Color-coded bounding boxes for different object classes
- 📊 Live FPS counter
- 📈 Object count display
- ⚙️ Adjustable confidence and IoU thresholds
- 💾 Option to save processed output video

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## 📂 Project Structure

```
.
├── obj_detect.py
├── yolov8n.pt
├── street.mp4
├── bus.jpg
├── detection_output.mp4 (Generated)
└── README.md
```

---

## 📥 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/yolov8-object-detection.git
```

Move into the project directory

```bash
cd yolov8-object-detection
```

Install the required libraries

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Running the Project

Run

```bash
python obj_detect.py
```

Select the input source

```
1. street.mp4
2. bus.jpg
3. webcam
```

---

## ⚙️ Optional Arguments

| Argument | Description |
|----------|-------------|
| `--model` | YOLO model (default: yolov8n.pt) |
| `--conf` | Confidence threshold |
| `--iou` | IoU threshold |
| `--img-size` | Inference image size |
| `--classes` | Detect only selected classes |
| `--save` | Save output video |


## 🎯 Detected Classes

The project currently highlights classes including:

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck
- Cat
- Dog

Additional classes supported by YOLOv8 can also be detected.

---

## 📊 Output

The application displays:

- Bounding boxes
- Class labels
- Confidence scores
- Live FPS
- Total detected objects

---

## 🔮 Future Improvements

- Object Tracking (ByteTrack/DeepSORT)
- Custom YOLOv8 model training
- GPU acceleration (CUDA)
- ROI-based detection
- Distance estimation
- Drone integration
- ROS2 integration




## 👨‍💻 Author

**Manav Subudhi**

Automation & Robotics Engineering Student

Interested in:
- Computer Vision
- Robotics
- Drones
- AI
- ROS2

