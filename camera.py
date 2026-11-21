import cv2
import time
import os
import easyocr
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
reader = easyocr.Reader(['en'], gpu=False)

if not os.path.exists("AI_Security_Alerts"):
    os.makedirs("AI_Security_Alerts")

cap = cv2.VideoCapture(0)
last_saved_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, conf=0.75)
    detected_any = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            coords = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            cls = int(box.cls.item())
            name = model.names[cls]
            conf = round(float(box.conf.item()) * 100, 1)

            detected_any = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ALERT: {name.upper()} {conf}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if name in ["car", "motorcycle"]:
                plate_crop = frame[int(y1 + (y2-y1)*0.4):y2, x1:x2]
                if plate_crop.size > 0:
                    ocr_result = reader.readtext(plate_crop)
                    if ocr_result:
                        plate_text = ocr_result[0][1]
                        cv2.putText(frame, f"PLATE: {plate_text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        now = datetime.now()
                        current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                        with open("AI_Security_Alerts/vehicle_logs.txt", "a") as f:
                            f.write(f"[{current_date_time}] Detected: {name.upper()} | Plate: {plate_text}\n")

    if detected_any:
        current_time = time.time()
        if current_time - last_saved_time > 3:
            photo_name = f"AI_Security_Alerts/alert_{int(current_time)}.png"
            cv2.imwrite(photo_name, frame)
            last_saved_time = current_time

    cv2.imshow("AI Smart Security Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()