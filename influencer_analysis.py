import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from urllib.request import urlretrieve
import uuid

# Step 1: Download Video from URL
def download_video(video_url, output_dir="videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_filename = os.path.join(output_dir, video_url.split("/")[-1])
    if not os.path.exists(video_filename):
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

# Step 3: Filter Small Faces
def filter_small_faces(face_locations, min_size=50):
    """Filter out faces with bounding boxes smaller than min_size."""
    filtered_faces = []
    for (top, right, bottom, left) in face_locations:
        if (bottom - top) > min_size and (right - left) > min_size:
            filtered_faces.append((top, right, bottom, left))
    return filtered_faces

# Step 4: Exclude Faces on Objects (Optional Advanced Step)
def filter_faces_on_objects(frame, face_locations, object_classes=["photo frame", "keychain"]):
    """
    Exclude faces overlapping with objects (e.g., photo frames, keychains).
    Requires a pre-trained object detection model.
    """
    # Placeholder: Replace with actual object detection logic
    # detected_objects = detect_objects(frame)
    # Assuming detected_objects = [{"class": "photo frame", "bbox": (top, right, bottom, left)}, ...]

    # For simplicity, skip this step in implementation unless integrated with object detection.
    return face_locations

# Step 5: Detect and Save Valid Faces
def detect_and_save_faces(frames, output_dir="Plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embeddings = []
    for frame_number, frame in enumerate(frames):
        face_locations = face_recognition.face_locations(frame)
        face_locations = filter_small_faces(face_locations)  # Filter out small faces
        face_locations = filter_faces_on_objects(frame, face_locations)  # Exclude faces on objects
        
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            face = frame[top:bottom, left:right]
            unique_id = uuid.uuid4().hex  # Generate a unique identifier
            filename = f"influencer_face_{frame_number}_{unique_id}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Save face
            print(f"Saved face to {filepath}")
            
            # Generate embeddings for clustering
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
            embeddings.append(face_encoding)
    return embeddings

# Step 6: Cluster Faces
def cluster_faces(embeddings):
    clustering_model = DBSCAN(metric='euclidean', eps=0.5, min_samples=5)
    labels = clustering_model.fit_predict(embeddings)
    return labels

# Step 7: Calculate Performance Metrics for Influencers
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

    all_embeddings = []
    video_to_clusters = {}

    # Process each video
    for idx, video_url in enumerate(video_urls):
        print(f"Processing video {idx + 1}/{len(video_urls)}...")
        video_path = download_video(video_url)
        frames = extract_frames(video_path)
        embeddings = detect_and_save_faces(frames)

        if embeddings:
            all_embeddings.extend(embeddings)
            video_to_clusters[video_url] = embeddings

    # Cluster embeddings to find unique influencers
    print("Clustering embeddings to identify unique influencers...")
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        labels = cluster_faces(all_embeddings)

        # Map video URLs to influencer clusters
        for video_url, embeddings in video_to_clusters.items():
            video_to_clusters[video_url] = [labels[i] for i in range(len(embeddings))]

    # Calculate influencer performance
    influencer_performance = calculate_influencer_performance(video_to_clusters, video_performance)

    # Display the results
    print("\nInfluencer Performance Metrics:")
    for influencer, avg_performance in influencer_performance.items():
        print(f"Influencer {influencer}: Average Performance = {avg_performance:.2f}")
