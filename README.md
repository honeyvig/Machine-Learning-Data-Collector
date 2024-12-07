# Machine-Learning-Data-Collector
a diligent and detail-oriented Machine Learning Data Collector to assist in building a robust dataset for training models in an advanced image recognition system. The role will involve capturing a diverse and comprehensive set of high-quality images of drones and guns that will be used in a demo showcasing our detection capabilities.

Data collection part
1) Match Models: Use drones and guns that match the models you’ll showcase in the demo.

2) Resolution & Quantity: Collect 10,000 images per class at 1080p resolution (1920x1080) camera.

3) Variety in Positioning: Capture each object (drone or gun) in different positions and orientations across images. object doesn't have to be in the center of every photo

4) Background Diversity: Change backgrounds frequently—use various locations and environments.

5) Lighting & Distance: Collect images in different lighting (day, night, indoor) and at varied distances for scale diversity.
6) ========================
To build a robust dataset for training models in an advanced image recognition system, we can use Python along with libraries like OpenCV and possibly some custom scripts to automate the image collection process based on the specifications you provided.

This task involves capturing high-quality images of drones and guns across various conditions. Below is a Python script that focuses on the collection part of this project. This code snippet will guide you through capturing and managing images using a camera.
Prerequisites:

    Install opencv-python for image capture and manipulation.
    A camera capable of capturing images at 1080p resolution.
    Access to drones and guns that match the models to showcase.
    Various locations and environments for background diversity.

Install OpenCV:

pip install opencv-python

Python Code for Image Collection:

import cv2
import os
import random
import time
from datetime import datetime

# Directories to save captured images
DRONE_DIR = 'data/drone_images/'
GUN_DIR = 'data/gun_images/'

# Create directories if they don't exist
os.makedirs(DRONE_DIR, exist_ok=True)
os.makedirs(GUN_DIR, exist_ok=True)

# Constants for capturing
IMAGE_COUNT = 10000  # Number of images to collect per class
IMAGE_WIDTH = 1920  # 1080p resolution width
IMAGE_HEIGHT = 1080  # 1080p resolution height

# Initialize the camera
camera = cv2.VideoCapture(0)  # Assuming the default webcam is used
camera.set(3, IMAGE_WIDTH)  # Set frame width (1920)
camera.set(4, IMAGE_HEIGHT)  # Set frame height (1080)

# Helper function to capture and save images
def capture_image(class_name, count, position):
    """
    Captures an image from the camera and saves it.
    class_name: Class of the object (drone/gun).
    count: The number of images taken so far for the class.
    position: Randomly varied position for diversity.
    """
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        return
    
    # Add some random variation to the captured image (random crop, angle)
    rotated_frame = rotate_frame(frame, position)
    file_name = f"{class_name}_{count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    # Save the image to the corresponding class folder
    if class_name == 'drone':
        cv2.imwrite(os.path.join(DRONE_DIR, file_name), rotated_frame)
    elif class_name == 'gun':
        cv2.imwrite(os.path.join(GUN_DIR, file_name), rotated_frame)
    
    print(f"Captured image: {file_name}")

# Function to simulate a random rotation or position change
def rotate_frame(frame, position):
    """
    This function simulates rotating the frame for capturing objects in different orientations.
    """
    angle = random.randint(-30, 30)  # Random rotation angle between -30 to 30 degrees
    rows, cols, _ = frame.shape
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_frame = cv2.warpAffine(frame, matrix, (cols, rows))
    
    # Adding random crop to simulate varying object positions
    x_offset = random.randint(0, 50)
    y_offset = random.randint(0, 50)
    cropped_frame = rotated_frame[y_offset:rows - y_offset, x_offset:cols - x_offset]
    
    return cropped_frame

# Function to simulate background diversity (this part can be done manually or by overlaying background images)
def change_background(frame):
    """
    In a real-world scenario, this would dynamically swap backgrounds (e.g., using a green screen).
    Here, we simulate it by just modifying the contrast and lighting for variety.
    """
    # Apply random changes to lighting/contrast
    contrast_factor = random.uniform(0.8, 1.2)
    brightness_factor = random.randint(-50, 50)
    modified_frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=brightness_factor)
    return modified_frame

# Main loop for data collection
def collect_data():
    print("Starting image collection process...")

    # Count for each class (drone/gun)
    drone_count = 0
    gun_count = 0
    
    # Loop for capturing images for both drone and gun classes
    while drone_count < IMAGE_COUNT or gun_count < IMAGE_COUNT:
        # Alternate between drone and gun for diversity
        if drone_count < IMAGE_COUNT:
            capture_image('drone', drone_count + 1, random.randint(0, 360))
            drone_count += 1
            time.sleep(random.uniform(1, 3))  # Add some delay between captures
        
        if gun_count < IMAGE_COUNT:
            capture_image('gun', gun_count + 1, random.randint(0, 360))
            gun_count += 1
            time.sleep(random.uniform(1, 3))  # Add some delay between captures
        
        # You can adjust the rate or add manual intervention to control when you want to change the background
        if drone_count % 100 == 0 or gun_count % 100 == 0:
            print(f"Captured {drone_count} drone images and {gun_count} gun images.")

    print("Image collection completed.")

# Start the data collection
collect_data()

# Release the camera once done
camera.release()
cv2.destroyAllWindows()

Explanation:

    Directory Setup:
        Two directories are created: drone_images and gun_images, to store the respective class images.

    Image Capture:
        The camera captures images at 1080p resolution (1920x1080) using OpenCV. The camera is set up using cv2.VideoCapture(0) to use the default webcam.
        Every captured image is stored with a timestamp in its file name for uniqueness.

    Rotation and Positioning:
        The rotate_frame function applies random rotations to the image to simulate different orientations and variations in the positioning of the objects.

    Background Diversity:
        The change_background function modifies lighting and contrast to simulate different backgrounds. This can be expanded to add more complex background replacement using techniques like chroma key (green screen) if available.

    Image Collection Loop:
        The collect_data function runs in a loop until 10,000 images are captured for both drone and gun classes. The loop alternates between capturing drone and gun images.
        A small delay (time.sleep()) is added to avoid too rapid image collection and simulate natural capturing intervals.

    Real-Time Logging:
        Logs are printed to track how many images have been captured for each class.

Customizations:

    Background Diversity:
        For more advanced background diversity, you can add a system to overlay images with different environments or use techniques such as green-screen and background removal.

    Image Quality:
        The camera settings and lighting conditions can be adjusted manually or via additional code to handle different environmental settings like day, night, indoor lighting, etc.

    Task Scheduling:
        You can modify the frequency of capturing images and schedule breaks to give time for manual intervention or object repositioning.

Next Steps:

    Collect images for both drones and guns according to the above setup.
    Once the images are captured, they can be preprocessed for training your image recognition model, e.g., resizing, normalization, and augmentations.
