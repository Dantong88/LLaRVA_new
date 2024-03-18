from moviepy.editor import ImageSequenceClip
import os

# Path to the folder containing your images
image_folder = '/home/niudt/LLaVA/process_dataset/eval/visualizations/demo_trace'  # Update this path
# Output video file path
output_video = '/home/niudt/LLaVA/process_dataset/eval/visualizations/demo_trace.mp4'  # Update this path

# Assuming your images are named in a way that sorting them alphabetically
# makes them in the correct order (e.g., frame1.jpg, frame2.jpg, ...)
# You may need to adjust the sorting logic based on your filenames
images = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

# Create a video clip
clip = ImageSequenceClip(images, fps=5)  # fps = frames per second

# Write the video file
clip.write_videofile(output_video)
