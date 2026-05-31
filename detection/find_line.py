import cv2, sys
VIDEO = sys.argv[1]
cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()
print(f"Video size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
cv2.imwrite("first_frame.jpg", frame)
print("Saved first_frame.jpg - open it and find the door y-coordinate")
