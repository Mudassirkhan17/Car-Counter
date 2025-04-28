import cv2
import numpy as np
import json
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Draw a counting line on a video and save the coordinates')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', '-o', default='counting_line.json', help='Output JSON file for the line coordinates')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview the line after drawing')
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return
    
    # Open the video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video_path}'.")
        return
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {width}x{height}")
    
    # Read first frame for drawing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
        return
    
    # Variables for line drawing
    line_points = []
    drawing = False
    
    # Function to handle mouse events
    def draw_line(event, x, y, flags, param):
        nonlocal line_points, drawing, frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(line_points) < 2:
                line_points.append((x, y))
                print(f"Point {len(line_points)} set at ({x}, {y})")
                
                # Draw the point
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
                if len(line_points) == 2:
                    # Draw the complete line
                    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)
                    print(f"Line drawn from {line_points[0]} to {line_points[1]}")
    
    # Create window and set mouse callback
    window_name = "Draw Counting Line - Press 'c' to clear, 's' to save, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_line)
    
    # Main loop
    while True:
        # Show the frame with drawn elements
        cv2.imshow(window_name, frame)
        
        # Get keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press 'c' to clear and start over
        elif key == ord('c'):
            line_points = []
            ret, frame = cap.read()
            if not ret:
                # If we can't read the frame, rewind to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            print("Line cleared. Draw a new line.")
        
        # Press 's' to save coordinates
        elif key == ord('s') and len(line_points) == 2:
            # Convert to list format for counting line
            limits = [line_points[0][0], line_points[0][1], 
                     line_points[1][0], line_points[1][1]]
            
            # Save to JSON file
            with open(args.output, 'w') as f:
                json.dump({
                    'limits': limits,
                    'video_width': width,
                    'video_height': height,
                    'points': line_points
                }, f, indent=4)
                
            print(f"Line coordinates saved to '{args.output}'")
            print(f"Use these coordinates in your code: limits = {limits}")
            
            # Preview if requested
            if args.preview:
                preview_line(args.video_path, limits)
                # After preview, get back to drawing mode
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                # Redraw the current line
                if len(line_points) == 2:
                    cv2.circle(frame, line_points[0], 5, (0, 0, 255), -1)
                    cv2.circle(frame, line_points[1], 5, (0, 0, 255), -1)
                    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def preview_line(video_path, limits):
    """
    Preview the counting line on the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video for preview")
        return
    
    print("Previewing line. Press 'q' to exit preview mode.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue
        
        # Draw the counting line
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
        
        # Display frame number
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show coordinates
        cv2.putText(frame, f"Line: {limits}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Line Preview - Press 'q' to exit", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 