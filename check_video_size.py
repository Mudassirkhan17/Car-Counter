import cv2
import os

def get_video_dimensions(video_path):
    """
    Get the dimensions of a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (width, height) of the video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture
    cap.release()
    
    return width, height, total_frames, fps

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define video path
    video_path = os.path.join(script_dir, "location1.MTS")  # Update this to your video filename
    
    try:
        width, height, total_frames, fps = get_video_dimensions(video_path)
        print(f"Video dimensions: {width}x{height}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        
        # Show a frame to verify
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            cv2.imshow("First Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 