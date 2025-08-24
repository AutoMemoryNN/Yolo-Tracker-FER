import cv2
import time
from core.detectors.face_detector import FaceDetector
from core.detectors.emotion_detector import EmotionDetector


def main():
    cap = cv2.VideoCapture(0)  # webcam
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector(backend="mtcnn")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Time face detection
        start_time = time.time()
        boxes = face_detector.detect(frame)
        face_detection_time = time.time() - start_time

        total_emotion_time = 0
        face_count = 0

        for x1, y1, x2, y2 in boxes:
            # Calculate center and size for square bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Use the larger dimension to create a square
            size = max(width, height)
            half_size = size / 2

            # Apply padding
            padding = size * 0.0
            padded_half_size = half_size + padding

            # Calculate new square coordinates
            x1 = max(0, int(center_x - padded_half_size))
            y1 = max(0, int(center_y - padded_half_size))
            x2 = min(frame.shape[1], int(center_x + padded_half_size))
            y2 = min(frame.shape[0], int(center_y + padded_half_size))

            face_roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

            result = {"emotion": None, "score": 0.0}
            if face_roi.size > 0:
                # Time emotion detection
                emotion_start_time = time.time()
                emotion_result = emotion_detector.detector.top_emotion(face_roi)
                emotion_time = time.time() - emotion_start_time
                total_emotion_time += emotion_time
                face_count += 1

                # TODO: REMOVE THIS SHIT
                if emotion_result:
                    emotion, score = emotion_result
                    # If top emotion is fear, get all emotions to find alternative
                    if emotion == "fear":
                        all_emotions = emotion_detector.detector.detect_emotions(
                            face_roi
                        )
                        if all_emotions and len(all_emotions) > 0:
                            # Sort emotions by score and get the second highest that's not fear
                            emotions_list = all_emotions[0]["emotions"]
                            sorted_emotions = sorted(
                                emotions_list.items(), key=lambda x: x[1], reverse=True
                            )
                            for emo, sc in sorted_emotions:
                                if emo != "fear":
                                    result = (emo, sc)
                                    break
                            else:
                                result = (emotion, score)
                        else:
                            result = (emotion, score)
                    else:
                        result = (emotion, score)
                else:
                    result = {"emotion": None, "score": 0.0}

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if result and result[0] is not None:
                emotion, score = result
                text = f"{emotion} ({score:.2f})"
                cv2.putText(
                    frame,
                    text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        # Display timing information
        timing_text = f"Face detection: {face_detection_time * 1000:.1f}ms"
        cv2.putText(
            frame,
            timing_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if face_count > 0:
            avg_emotion_time = total_emotion_time / face_count
            emotion_timing_text = f"Avg emotion: {avg_emotion_time * 1000:.1f}ms"
            cv2.putText(
                frame,
                emotion_timing_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Face + Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
