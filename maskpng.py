# This script is used to create a mask from a polygon
# The mask is a black image with a white polygon where we want to detect vehicles
# The polygon is defined by the points in the mask.png file
# The mask.png file is a black image with a white polygon where we want to detect vehicles
# The polygon is defined by the points in the mask.png file
# The mask.png file is a black image with a white polygon where we want to detect vehicles
import cv2
import numpy as np
import os

def create_mask_from_polygon(video_path, output_path):
    """
    Create a mask from a polygon defined in an image, using the first frame of a video.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the mask will be saved
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read frame from video")
    
    # Release the video capture
    cap.release()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No polygon found in the image")
    
    # Create a black mask
    mask = np.zeros_like(gray)
    
    # Draw the largest contour (polygon) in white
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    
    # Save the mask
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    video_path = os.path.join(script_dir, "location1.MTS")  # Update this to your video filename
    output_path = os.path.join(script_dir, "mask_output.png")
    
    try:
        create_mask_from_polygon(video_path, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


