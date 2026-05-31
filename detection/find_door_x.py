import cv2, sys

VIDEO = sys.argv[1]
cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

# Draw several vertical lines so you can pick the right one
for x, color in [(400,(255,0,0)), (530,(0,0,255)), (650,(0,255,0)), (780,(255,165,0))]:
    cv2.line(frame, (x,0), (x,frame.shape[0]), color, 2)
    cv2.putText(frame, f"x={x}", (x+5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imwrite("door_lines.jpg", frame)
print("Saved door_lines.jpg — find which vertical line aligns with the glass door edge")
