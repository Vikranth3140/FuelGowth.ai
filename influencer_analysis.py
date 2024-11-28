import cv2
import face_recognition
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from urllib.request import urlretrieve
import os

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

# Step 3: Detect Faces in Frames and Generate Embeddings
def detect_faces_and_embeddings(frames):
    embeddings = []
    for frame in frames:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        embeddings.extend(face_encodings)
    return embeddings

# Step 4: Cluster Faces to Identify Unique Influencers
def cluster_faces(embeddings):
    clustering_model = DBSCAN(metric='euclidean', eps=0.5, min_samples=5)
    labels = clustering_model.fit_predict(embeddings)
    return labels

# Step 5: Process Video URLs to Extract Unique Influencers
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

# Step 6: Calculate Performance Metrics for Each Influencer
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

# Main Script
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
