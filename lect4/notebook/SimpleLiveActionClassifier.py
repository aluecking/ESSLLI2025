# Simplified Live Action Classification Demo
# --------------------------------------------------------------------------------
# ▸  pip install opencv-python mmcv mmpose==1.3.0 scikit-learn joblib pandas numpy
# ▸  place your trained model (joblib) & feature‑extractor in the same folder
# --------------------------------------------------------------------------------

import cv2
import time
import joblib
import numpy as np
import pandas as pd
from mmpose.apis import MMPoseInferencer
from HandPoseFeatureGenerator import HandPoseFeatureGenerator


class SimpleLiveActionClassifier:
    def __init__(self, model_path='data/models/feedforward_model.joblib', webcam_idx=0):
        # Initialize components
        self.feature_extractor = HandPoseFeatureGenerator()
        self.inferencer = MMPoseInferencer('hand')
        self.clf = joblib.load(model_path)

        # Camera setup
        self.cap = cv2.VideoCapture(webcam_idx, cv2.CAP_DSHOW)
        assert self.cap.isOpened(), "Cannot open webcam"

        # Performance tracking
        self.fps_counter = 0.
        self.fps_start_time = time.time()
        self.fps = 0.

        print("Simplified Live Action Classifier initialized")
        print("Press [q] to quit")

    def extract_pose_features(self, frame):
        """Extract hand pose features from a single frame"""

        # Run pose inference
        result = next(self.inferencer(frame))
        # Handle different result structures
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']

            if predictions and len(predictions) > 0:
                pose_data = {'instances': predictions[0]}
            else:
                pose_data = {'instances': []}
        else:
            pose_data = {'instances': []}
        # Add frame_id for compatibility with feature extractor
        pose_data['frame_id'] = 0

        # Extract features
        features = self.feature_extractor.extract_features_with_defaults(pose_data)
        return features

    def classify_features(self, features):
        """Classify action from extracted features"""
        try:
            # Check if features are empty (all values are 0)
            finger_features = ['thumb_extension', 'index_extension', 'middle_extension',
                               'ring_extension', 'pinky_extension']

            if not features or all(features.get(feat, 0) == 0 for feat in finger_features):
                return "Hands not detected"

            # Convert to DataFrame
            features_df = pd.DataFrame([features])

            # Fill NaN values with 0
            features_df = features_df.fillna(0)

            # Make prediction
            prediction = self.clf.predict(features_df)[0]

            # Get prediction probability for confidence
            if hasattr(self.clf, 'predict_proba'):
                proba = self.clf.predict_proba(features_df)[0]
                confidence = np.max(proba)
                return f"{prediction} ({confidence:.2f})"
            else:
                return str(prediction)

        except Exception as e:
            print(f"Classification error: {e}")
            return "Classification error"

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            self.fps = 30 / (current_time - self.fps_start_time + 1e-6)
            self.fps_start_time = current_time

    def create_visualization(self, frame, label):
        """Create visualization with overlays"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Semi-transparent overlay for text background
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Main prediction label
        label_text = f"Action: {label}"
        cv2.putText(vis, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS info
        info_text = f"FPS: {self.fps:.1f}"
        cv2.putText(vis, info_text, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Instructions
        instructions = "Press 'q' to quit"
        cv2.putText(vis, instructions, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def run(self):
        """Main execution loop"""
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Extract features from current frame
                features = self.extract_pose_features(frame)

                # Classify action
                label = self.classify_features(features)

                # Create visualization
                vis_frame = self.create_visualization(frame, label)

                # Display
                cv2.imshow("Live Action Classification", vis_frame)

                # Update FPS
                self.update_fps()

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    classifier = SimpleLiveActionClassifier()
    classifier.run()
