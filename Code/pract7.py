import cv2
import numpy as np

video_path = './Media/input_video.mov'
cap = cv2.VideoCapture(video_path)

# Implementing Sparse Optical Flow (Lucas-Kanade Method)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Creating some random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Shi-Tomasi Corner Detector - selecting the pixels to track
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Creating a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tracking Specific Objects
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Lucas-Kanade: Sparse Optical Flow
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Visualizing video
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame, mask)

    cv2.imshow('Optical Flow', img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()

# Implementing Dense Optical Flow (Horn-Schunk method)

# Read the first frame
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Define Horn-Schunck parameters
alpha = 1.0
iterations = 100

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate spatial derivatives of the two frames
    Ix, Iy = np.gradient(prvs)
    It = next - prvs

    # Initialize the flow vectors to zero
    u = np.zeros_like(prvs)
    v = np.zeros_like(prvs)

    # Iteratively update the flow vectors
    for _ in range(iterations):
        u_avg = cv2.filter2D(u, -1, np.array([[0.25, 0.5, 0.25], [0.5, 0, 0.5], [0.25, 0.5, 0.25]]))
        v_avg = cv2.filter2D(v, -1, np.array([[0.25, 0.5, 0.25], [0.5, 0, 0.5], [0.25, 0.5, 0.25]]))
        # Update the flow vectors
        u = u_avg - (Ix * (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2))
        v = v_avg - (Iy * (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2))

    # Visualize the dense optical flow (optional)
    flow_visualization = np.zeros((prvs.shape[0], prvs.shape[1], 3), dtype=np.uint8)
    flow_visualization[..., 0] = u * 10 + 128
    flow_visualization[..., 1] = v * 10 + 128
    flow_visualization[..., 2] = 255

    # Display the dense optical flow
    cv2.imshow('Dense Optical Flow (Horn-Schunck)', flow_visualization)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Set the current frame as the previous frame for the next iteration
    prvs = next

cap.release()
cv2.destroyAllWindows()