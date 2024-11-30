import pandas as pd

# Example influencer data
influencer_data = {
    0: {"average_performance": 0.97, "example_image": "Plots/influencer_face_0_c40118c19b6242bfb4b0ed80154f022d.png"},
    1: {"average_performance": 0.98, "example_image": "Plots/influencer_face_1_f24a2446da9f40acb831a5abf23edad4.png"},
    2: {"average_performance": 0.75, "example_image": "Plots/influencer_face_2_23a7bd2d91554f77a959615885984091.png"},
    3: {"average_performance": 0.75, "example_image": "Plots/influencer_face_3_d14768b498f2434c9cb99606a04b3f15.png"},
    4: {"average_performance": 0.74, "example_image": "Plots/influencer_face_4_ac22de7678704d8888d880e79afd4f5d.png"},
}

# Prepare simplified data for the CSV
csv_data = [
    {
        "Influencer ID": influencer_id,
        "Average Performance": data["average_performance"],
        "Face Image Path": data["example_image"]
    }
    for influencer_id, data in influencer_data.items()
]

# Convert to DataFrame
df_csv = pd.DataFrame(csv_data)

# Save to CSV
output_csv_path = "influencer_performance_summary.csv"
df_csv.to_csv(output_csv_path, index=False)
print(f"CSV file saved at: {output_csv_path}")