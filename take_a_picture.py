import cv2
import os
import time

cap = cv2.VideoCapture(2)

save_dir = "/home/ohheemin/dataset2/train/50c90d"
os.makedirs(save_dir, exist_ok=True)

file_prefix = "train" 

save_interval = 0.2
last_saved_time = time.time()
frame_count = 1  # 120번부터 시작

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Video Stream", frame)
    current_time = time.time()

    if current_time - last_saved_time >= save_interval:
        filename = os.path.join(save_dir, f"{file_prefix}_{frame_count:04d}+12.jpg")
        cv2.imwrite(filename, frame)
        print(f"{filename} 저장됨")
        last_saved_time = current_time
        frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
