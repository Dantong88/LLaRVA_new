from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Replace 'your_image.jpg' with the path to your image
image_path = './87212_62.jpg'
image = Image.open(image_path)

# Example list of points as (x, y) tuples
points = 'pick [214, 221, 1] blue [183, 218, 1] chip [185, 217, 1] bag [224, 227, 1] from [241, 237, 1] bottom [239, 240, 1] drawer [281, 229, 1] and [292, 183, 1] place [282, 104, 1] on [264, 65, 1] counter [269, 71, 1]'  # Replace these with your points

_points = []
# points = [(280, 117), (220, 194), (185, 221), (167, 230), (155, 234), (164, 228,), (182, 219), (192, 183), (238, 111), (241, 64), (232, 75) ]
points = [(158, 156)]
# Create a drawing context
draw = ImageDraw.Draw(image)

# Define point properties
point_radius = 5
point_color = 'red'

# Plot each point on the image
for point in points:
    x, y = point
    # Draw a small circle around each point
    draw.ellipse((x-point_radius, y-point_radius, x+point_radius, y+point_radius), fill=point_color)

# Save the modified image
image.save('./87212_8_results.jpg')

# Optionally display the modified image
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
