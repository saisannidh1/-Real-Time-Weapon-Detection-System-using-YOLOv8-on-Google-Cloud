# 🔫 Real-Time Weapon Detection System using YOLOv8 on Google Cloud

## 📌 Overview

This project is a real-time weapon detection system that uses the power of YOLOv8 for fast and accurate object detection. It combines local webcam-based surveillance with cloud-hosted AI models on Google Cloud Platform (GCP), providing a scalable and self-learning system for public safety applications.

## 🎯 Key Features

* 🧠 **YOLOv8-based weapon detection**
* 🌐 **Live webcam feed from local machine**
* ☁️ **Cloud-hosted detection model on GCP VM**
* 📄 **Auto-upload of detected frames to GCS bucket**
* ♻️ **Self-training model that retrains on new data**
* 🗂️ **Full-frame saving with YOLO-format annotations**
* 🔐 **Designed for security and smart surveillance use cases**

---

## 🧱 Architecture

```
+----------------+        Live Feed        +-----------------+
|  Local Machine |  -------------------->  | GCP VM Instance |
|  (with camera) |                        |  (YOLOv8 Model)  |
+-------+--------+                        +--------+--------+
        |                                          |
        | On Detection                             |
        |  - Save full frame                       |
        |  - Save YOLO label                       |
        |  - Upload to GCS bucket                  |
        |                                          |
        +-------------------> Trigger Training ----+
```

---

## 🛠️ Tech Stack

| Component        | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Object Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) |
| Local App        | Python, OpenCV, Flask (optional)                     |
| Cloud Compute    | Google Cloud VM (Ubuntu, PyTorch)                    |
| Storage          | Google Cloud Storage (GCS)                           |
| Automation       | Python script to trigger training                    |
| Dataset Format   | YOLO `.txt` with full-frame bounding boxes           |

---

## ⚙️ Setup Instructions

### 🔧 Prerequisites

* Google Cloud Platform account
* GCP VM instance with:

  * Python 3.x
  * YOLOv8 (Ultralytics)
  * Google Cloud SDK installed
* GCS bucket mounted to VM
* Webcam-enabled local system

---

### 💻 Local Machine Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/weapon-detection-yolov8-gcp.git
cd weapon-detection-yolov8-gcp
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the local application:

```bash
python local_app.py
```

> This will open your camera, stream frames to the cloud, and display detections in real-time.

---

### ☁️ Google Cloud VM Setup

1. SSH into your GCP VM instance.

2. Activate Python environment:

```bash
source venv/bin/activate
```

3. Start the detection server:

```bash
python cloud_server.py
```

4. Configure GCS bucket and training script:

   * Upload frames and `.txt` labels to the `dataset/` directory in the bucket.
   * A `watchdog` or GCS trigger runs `train.py` whenever new data is added.

---

## 🦪 Training the Model

* Training is triggered automatically when new labeled data is uploaded.
* The model is trained using the YOLOv8 CLI or API with the following structure:

```bash
yolo task=detect mode=train model=best.pt data=data.yaml epochs=50 imgsz=640
```

---

## 📝 Dataset Format

Each detection result is stored as:

```
/dataset/
  ├— images/
  │   └— frame123.jpg
  └— labels/
      └— frame123.txt  # YOLO format: <class_id> <x_center> <y_center> <width> <height>
```

---

## 🚀 Deployment Flow

1. Start VM on GCP.
2. Run the detection server in VM.
3. Run local client and allow access to your webcam.
4. Detected weapons:

   * Save full frame
   * Save YOLO label
   * Upload to GCS
   * Trigger training on cloud

---

## 📊 Future Improvements

* Multi-class detection (person + weapon)
* Add real-time dashboard with alerts
* Optimize model retraining with active learning
* Convert to REST API-based microservices
* Dockerize the entire system for portability

---

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contribution

Contributions, issues, and feature requests are welcome!
Feel free to fork and submit a pull request.

---

## 🤛 Author

**Sai Sannidh**

[GitHub](https://github.com/saisannidh535354)
