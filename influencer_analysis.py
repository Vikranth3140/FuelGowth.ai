import cv2
import face_recognition
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from urllib.request import urlretrieve
import os
import matplotlib.pyplot as plt
import uuid

# Step 1: Download Video from URL
def download_video(video_url, output_dir="videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_filename = os.path.join(output_dir, video_url.split("/")[-1])
    urlretrieve(video_url, video_filename)
    return video_filename

# Step 2: Extract Frames from Videos
def extract_frames(video_path, frame_rate=30):
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    frames = []
    count = 0
    while success:
        if count % frame_rate == 0:  # Capture every nth frame
            frames.append(frame)
        success, frame = video.read()
        count += 1
    video.release()
    return frames

# Step 3: Save Detected Faces to Directory
def save_faces(frame, face_locations, output_dir="Plots", influencer_id=None, frame_number=None):
    """Save faces detected in a frame to a specified directory with unique filenames."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (top, right, bottom, left) in enumerate(face_locations):
        face = frame[top:bottom, left:right]
        # Generate a unique filename for each face
        unique_id = uuid.uuid4().hex  # Unique identifier
        filename = f"influencer_{influencer_id if influencer_id is not None else idx}_{frame_number}_{unique_id}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        print(f"Saved face to {filepath}")

# Step 4: Detect Faces and Save Embeddings
def detect_faces_and_embeddings(frames, output_dir="Plots"):
    embeddings = []
    for frame in frames:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        embeddings.extend(face_encodings)

        # Save detected faces
        save_faces(frame, face_locations, output_dir)
    return embeddings

# Step 5: Cluster Faces to Identify Unique Influencers
def cluster_faces(embeddings):
    clustering_model = DBSCAN(metric='euclidean', eps=0.5, min_samples=5)
    labels = clustering_model.fit_predict(embeddings)
    return labels

# Step 6: Process Videos and Extract Unique Influencers
def process_videos(video_urls):
    influencer_clusters = {}
    all_embeddings = []
    video_to_clusters = {}

    for idx, video_url in enumerate(video_urls):
        print(f"Processing video {idx + 1}/{len(video_urls)}...")
        video_path = download_video(video_url)
        frames = extract_frames(video_path)
        embeddings = detect_faces_and_embeddings(frames)

        if embeddings:
            all_embeddings.extend(embeddings)
            video_to_clusters[video_url] = embeddings

    # Cluster all embeddings to find unique influencers
    print("Clustering embeddings to identify unique influencers...")
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        labels = cluster_faces(all_embeddings)

        # Map video URLs to influencer clusters
        for video_url, embeddings in video_to_clusters.items():
            video_to_clusters[video_url] = [labels[i] for i in range(len(embeddings))]

        influencer_clusters = labels

    return influencer_clusters, video_to_clusters

# Step 7: Calculate Performance Metrics for Each Influencer
def calculate_influencer_performance(video_to_clusters, video_performance):
    influencer_performance = {}
    for video_url, clusters in video_to_clusters.items():
        performance = video_performance.get(video_url, 0)
        for cluster in set(clusters):
            if cluster not in influencer_performance:
                influencer_performance[cluster] = []
            influencer_performance[cluster].append(performance)

    # Average performance for each influencer
    for influencer, performances in influencer_performance.items():
        influencer_performance[influencer] = np.mean(performances)

    return influencer_performance

# Main Script to Process videos and extract unique influencers
if __name__ == "__main__":
    # Load video URLs and performance data
    data = pd.read_csv("dataset/data.csv")
    video_urls = data["Video URL"].tolist()
    video_performance = dict(zip(data["Video URL"], data["Performance"]))

    # Process videos and extract influencer clusters
    influencer_clusters, video_to_clusters = process_videos(video_urls)

    # Calculate influencer performance metrics
    influencer_performance = calculate_influencer_performance(video_to_clusters, video_performance)

    # Display the results
    print("\nInfluencer Performance Metrics:")
    for influencer, avg_performance in influencer_performance.items():
        print(f"Influencer {influencer}: Average Performance = {avg_performance:.2f}")
