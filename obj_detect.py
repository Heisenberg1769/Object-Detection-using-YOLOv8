

import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
print("YOLOv8 Object Detection, which one you want to use?")
a=print("1. street.mp4")
b=print("2. bus.jpg")
c=print("3. webcam")

x=int(input("enter the source: "))

if x==1:
    DEFAULT_SOURCE = "street.mp4"
elif x==2:
    DEFAULT_SOURCE = "bus.jpg"
elif x==3:
    DEFAULT_SOURCE = "0"


# Configuration 
DEFAULT_MODEL    = "yolov8n.pt"  
DEFAULT_SOURCE1  = "street.mp4"   
DEFAULT_SOURCE2  = "bus.jpg"  
DEFAULT_SOURCE2  = "0"  
DEFAULT_CONF     = 0.35
DEFAULT_IOU      = 0.45
DEFAULT_IMG_SIZE = 480            
DEFAULT_CLASSES  = None          
TARGET_CLASSES   = [0, 2, 3, 5, 7]  # person, car, truck, bus, motorcycle

CLASS_COLORS = {
    0:  (0, 255, 0),    # person   - green
    1:  (255, 0, 0),    # bicycle  - blue
    2:  (0, 0, 255),    # car      - red
    3:  (255, 165, 0),  # motorcycle - orange
    5:  (128, 0, 128),  # bus      - purple
    7:  (0, 255, 255),  # truck    - yellow
    15: (255, 20, 147), # cat      - pink
    16: (0, 128, 255),  # dog      - light blue
}


def get_color(class_id):
    return CLASS_COLORS.get(class_id, (200, 200, 200))


def draw_detections(frame, results, fps):
    """Draw bounding boxes, labels, confidence scores and FPS on frame."""
    det_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = result.names[cls]
            color = get_color(cls)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text  = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
            det_count += 1

    # HUD overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}",        (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {det_count}",  (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, det_count


def run_detection(source, model_path, conf, iou, img_size, classes, save_output):
    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output:
        out_path = "detection_output.mp4"
        writer = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (w, h))
        print(f"[INFO] Saving output to {out_path}")

    print("[INFO] Running detection — press Q to quit\n")
    fps       = 0.0
    frame_idx = 0
    t_prev    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        # Run inference
        results = model.predict(
          frame,
          device="cpu",
          conf=conf,
          iou=iou,
          classes=classes,
          imgsz=img_size
          )

        # FPS calculation (rolling)
        frame_idx += 1
        if frame_idx % 10 == 0:
            t_now = time.time()
            fps   = 10 / (t_now - t_prev + 1e-6)
            t_prev = t_now

        frame, _ = draw_detections(frame, results, fps)

        cv2.imshow("Object Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif x==2:
            time.sleep(5)  # Slow down for image display2

        
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")


#  Entry Point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--source",   default=DEFAULT_SOURCE,   help="video path or 0 for webcam")
    parser.add_argument("--model",    default=DEFAULT_MODEL,    help="yolov8n/s/m/l/x.pt")
    parser.add_argument("--conf",     default=DEFAULT_CONF,     type=float, help="confidence threshold")
    parser.add_argument("--iou",      default=DEFAULT_IOU,      type=float, help="IOU threshold")
    parser.add_argument("--img-size", default=DEFAULT_IMG_SIZE, type=int,   help="inference image size")
    parser.add_argument("--classes",  default=None, nargs="+",  type=int,   help="filter classes by ID")
    parser.add_argument("--save",     action="store_true",                  help="save output video")
    args = parser.parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
        classes=args.classes,
        save_output=args.save
    )