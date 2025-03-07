# Make sure you've already run the previous cells that define:
# - fpga_image_rgb (from your FPGA image)
# - labels_fpga_merged (from your segmentation)

import numpy as np
from skimage import color
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Create a synthetic dataset for demonstration
# This is a fallback in case fpga_image_rgb or labels_fpga_merged aren't defined
if 'fpga_image_rgb' not in globals() or 'labels_fpga_merged' not in globals():
    print("Creating synthetic data for demonstration...")
    # Create synthetic image and labels
    synthetic_image = np.random.rand(100, 100)
    synthetic_labels = np.zeros((100, 100), dtype=int)
    
    # Create a few regions
    synthetic_labels[10:30, 10:30] = 1
    synthetic_labels[40:70, 40:70] = 2
    synthetic_labels[20:50, 60:90] = 3
    
    # Use these instead of the FPGA data
    image_for_features = synthetic_image
    labels_for_features = synthetic_labels
else:
    print("Using FPGA image and segmentation from previous cells...")
    image_for_features = color.rgb2gray(fpga_image_rgb)
    labels_for_features = labels_fpga_merged

# Extract features from segmented regions
def extract_features(image, labels):
    # Get unique regions (excluding background)
    regions = np.unique(labels)
    regions = regions[regions > 0]
    
    features = []
    region_labels = []
    
    # For each region, extract features
    for region_id in regions:
        # Create mask for this region
        mask = labels == region_id
        
        # Skip small regions
        if np.sum(mask) < 100:
            continue
            
        # Extract region properties
        props = regionprops(mask.astype(int), image)
        if len(props) == 0:
            continue
            
        prop = props[0]
        
        # Extract region from original image
        region_img = image[mask]
        
        # Calculate features
        feature_vector = [
            prop.area,                      # Area
            prop.perimeter,                 # Perimeter
            prop.eccentricity,              # Eccentricity
            prop.mean_intensity,            # Mean intensity
            np.std(region_img),             # Standard deviation of intensity
            prop.major_axis_length,         # Major axis length
            prop.minor_axis_length,         # Minor axis length
            prop.solidity,                  # Solidity
        ]
        
        features.append(feature_vector)
        
        # For demonstration, assign random labels (0 or 1)
        region_labels.append(np.random.randint(0, 2))
    
    return np.array(features), np.array(region_labels)

# Extract features from our image
X, y = extract_features(image_for_features, labels_for_features)

print(f"Extracted features for {len(X)} regions")

# Simple NumPy-based train/test split (70% train, 30% test)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.7 * len(X))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Simple NumPy-based classifier (nearest centroid)
def train_nearest_centroid(X, y):
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(axis=0)
    return centroids

def predict_nearest_centroid(X, centroids):
    classes = list(centroids.keys())
    predictions = []
    for sample in X:
        distances = [np.linalg.norm(sample - centroids[c]) for c in classes]
        predictions.append(classes[np.argmin(distances)])
    return np.array(predictions)

# Train the model
centroids = train_nearest_centroid(X_train, y_train)

# Make predictions
y_pred = predict_nearest_centroid(X_test, centroids)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Visualize some results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7, label='True Labels')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='plasma', marker='x', alpha=0.5, label='Predictions')
plt.title("Feature Space: Area vs Perimeter")
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.legend()
plt.show()

# Feature importance (simple correlation-based approach)
feature_names = [
    'Area', 'Perimeter', 'Eccentricity', 'Mean Intensity',
    'Std Intensity', 'Major Axis Length', 'Minor Axis Length', 'Solidity'
]

# Calculate correlation of each feature with the target
importance = []
for i in range(X.shape[1]):
    corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
    importance.append(corr)

# Sort features by importance
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Correlation-based)")
plt.bar(range(X.shape[1]), [importance[i] for i in indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()