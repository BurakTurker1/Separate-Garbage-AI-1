import cv2
import time

# Select the camera device (default is the primary camera)
camera = cv2.VideoCapture(0)  # 0 represents the primary camera. You can use a different number for additional cameras.

# Capture consecutive frames for a duration of 3 seconds after the camera is opened
capture_duration = 3  # Specify how many seconds frames will be captured

# Time interval between each frame (e.g., 0.1 seconds)
frame_interval = 0.1  # Adjust the time between each frame

# Number of consecutive frames to capture
frame_count = int(capture_duration / frame_interval)

# Variable to store the last captured frame
last_frame = None

for i in range(frame_count):
    # Wait for the specified interval between each frame
    cv2.waitKey(int(frame_interval * 1000))  # Wait time in milliseconds

    # Capture the camera image
    ret, frame = camera.read()

    if ret:
        # Store the last captured frame
        last_frame = frame

# Save the last captured frame to the specified path
if last_frame is not None:
    save_path = 'clearlyPhoto.png'  # Specify the file name and path for saving
    cv2.imwrite(save_path, last_frame)
    print(f'The last captured photo has been successfully saved: {save_path}')

# Release the camera connection
camera.release()
cv2.destroyAllWindows()
