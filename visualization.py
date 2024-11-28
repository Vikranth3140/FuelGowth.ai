import matplotlib.pyplot as plt

# Data for visualization
influencers = [0, 1, 2, 3, 4, 5]
avg_performance = [0.98, 1.03, 1.03, 0.53, 0.31, 0.31]

# Create bar chart
plt.bar(influencers, avg_performance)
plt.xlabel('Influencers')
plt.ylabel('Average Performance')
plt.title('Influencer Performance Metrics')

# Save the chart as an image
output_path = "influencer_performance_metrics.png"
plt.savefig(output_path)
plt.close()
