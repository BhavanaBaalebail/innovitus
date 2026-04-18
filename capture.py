import cv2
import os

mode = input("Enter mode (n = notdrowsy, d = drowsy): ").strip()

if mode == "n":
    save_path = "data/train/notdrowsy"
elif mode == "d":
    save_path = "data/train/drowsy"
else:
    print("Invalid mode")
    exit()

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        filename = f"{save_path}/img_{count}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
