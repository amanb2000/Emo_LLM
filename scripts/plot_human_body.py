import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the image
img = mpimg.imread('human_body_outline.jpg')

# Create fake data
body_part_values = {
    'head': 0.1,
    'neck': 0.2,
    'chest': 0.7,
    'shoulder': 0.3,
    'upper_arm': 0.4,
    'forearm': 0.5,
    'hand': 0.6,
    'stomach': 0.8,
    'thigh': 0.5,
    'calf': 0.3,
    'foot': 0.1
}

# Define body parts and their positions
body_parts = {
    'head': (487, 132),
    'neck': (492, 272),
    'chest': (498, 507),
    'shoulder_left': (285, 390),
    'shoulder_right': (700, 390),
    'forearm_left': (226, 826),
    'forearm_right': (753, 788),
    'upper_arm_left': (274, 586),
    'upper_arm_right': (721, 586),
    'hand_left': (194, 975),
    'hand_right': (801, 964),
    'stomach': (498, 794),
    'thigh_left': (402, 1139),
    'thigh_right': (590, 1139),
    'calf_left': (407, 1533),
    'calf_right': (577, 1533),
    'foot_left': (407, 1868),
    'foot_right': (583, 1873)
}

# Create an overlay with the same size as the image
overlay = np.zeros((img.shape[0], img.shape[1], 4))

# Add Gaussian patches
for part, value in body_part_values.items():
    # Color from blue (0) to red (1)
    color = (1 - value, 0, value, 0.6)  # RGB with alpha channel

    if part in ['shoulder', 'upper_arm', 'forearm', 'hand', 'thigh', 'calf', 'foot']:
        for side in ['left', 'right']:
            key = f'{part}_{side}'
            if key in body_parts:
                print(f"Processing: {key}")
                pos = body_parts[key]
                y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                mask = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2.0 * 70.0**2))
                for c in range(3):  # Apply color to RGB channels
                    overlay[:, :, c] += mask * color[c]
                overlay[:, :, 3] = np.maximum(overlay[:, :, 3], mask * 0.7)  # Alpha channel
    else:
        key = part
        if key in body_parts:
            print(f"Processing: {key}")
            pos = body_parts[key]
            y, x = np.ogrid[:img.shape[0], :img.shape[1]]
            mask = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2.0 * 70.0**2))
            for c in range(3):  # Apply color to RGB channels
                overlay[:, :, c] += mask * color[c]
            overlay[:, :, 3] = np.maximum(overlay[:, :, 3], mask * 0.7)  # Alpha channel

# Ensure the overlay alpha does not exceed 1
overlay[:, :, 3] = np.clip(overlay[:, :, 3], 0, 1)

# Create the figure and axis
fig, ax = plt.subplots()

# Show the original image
ax.imshow(img)

# Overlay the Gaussian patches
ax.imshow(overlay, alpha=0.75)

ax.axis('off')
plt.show()

