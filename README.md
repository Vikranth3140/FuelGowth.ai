# Influencer Analytics System

## FuelGowth.ai Assignment

## 🚀 Overview
This project analyzes videos to identify unique influencers, calculate their average performance, and rank them based on their impact. The system uses **face detection**, **clustering**, and **performance metrics** to provide actionable insights for influencer marketing campaigns.

## 📂 Features
- Detects faces from videos and clusters them to identify unique influencers.
- Calculates and ranks influencers based on their average performance across videos.
- Saves detected faces for easy visualization and verification.
- Provides full traceability of detected faces to their source videos and frames.

## 📈 Future Enhancements
- Incremental analysis for handling new videos over time without reprocessing existing data.
- Advanced metrics like engagement rate, sentiment analysis, and retention time.
- Real-time video processing and cross-platform analytics.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Libraries:
  - `face_recognition`
  - `opencv-python`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `openpyxl`

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset
Prepare a CSV file with the following columns:
- `Video URL`: The URLs of the videos to process.
- `Performance`: A numerical value representing the performance of each video.

Example:
```csv
Video URL,Performance
https://example.com/video1.mp4,1.25
https://example.com/video2.mp4,2.10
```

---

## 🖥️ Usage

### 1. Run the Script
Run the main script to process videos and generate outputs:
```bash
python influencer_analysis.py
```

### 2. Outputs
The system generates:
- **Face Images**: Saved in the `Plots/` directory.
- **CSV File**: A file summarizing influencer performance and linking face images.

### 3. Example Output
```text
Processing video 1/100...
Saved face to Plots/influencer_face_0_abc123.png
Saved face to Plots/influencer_face_1_xyz456.png
Clustering embeddings to identify unique influencers...

Influencer Performance Metrics:
Influencer 0: Average Performance = 1.25
Influencer 1: Average Performance = 2.10
```

---

## 🧩 Project Structure
```
.
├── Plots/                     # Directory for saved face images
├── influencer_analysis.py     # Main script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── influencer_performance_summary.csv  # Output file with influencer metrics
```

---

## 📋 Limitations
1. Real-time processing is not implemented; the system processes videos in batches.
2. Object detection for filtering non-direct faces (e.g., faces on photo frames) is currently simplified.
3. Advanced metrics (e.g., engagement rate, sentiment analysis) are not included.

---

## 🌟 Future Work
1. **Incremental Updates**: Enable processing new videos incrementally.
2. **Real-Time Capabilities**: Add streaming support for real-time video analytics.
3. **Ethical Features**: Ensure compliance with privacy laws and implement face anonymization if required.
4. **Advanced Insights**: Include predictive analytics for influencer performance.