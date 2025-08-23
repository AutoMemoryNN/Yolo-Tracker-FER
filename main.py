import cv2
from huggingface_hub import hf_hub_download
from core.detectors.face_detector import FaceDetector

def main():
    cap = cv2.VideoCapture(1)  # webcam
    detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        cv2.imshow("Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
