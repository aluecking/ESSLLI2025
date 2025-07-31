import numpy as np
import json
from typing import List, Dict, Tuple, Optional


class HandPoseFeatureGenerator:
    """
    Feature generator for hand pose classification between eating and drinking actions.

    Features extracted:
    - Finger curl/extension patterns
    - Pinch detection
    - Finger configurations
    """

    def __init__(self):
        # Define keypoint indices based on standard hand pose models
        # Assuming 21 keypoints: wrist + 4 fingers * 5 joints each
        self.keypoint_mapping = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

    def extract_features(self, frame_data: Dict) -> Dict[str, float]:
        """
        Extract finger-specific features from a single frame of hand pose data.

        Args:
            frame_data: Dictionary containing frame_id and instances with keypoints

        Returns:
            Dictionary of extracted finger features
        """
        features = {}
        print(frame_data)
        print("=====================")
        if not frame_data.get('instances'):
            return self._get_empty_features()

        # Filter out instances where bbox_score == 1
        valid_instances = [inst for inst in frame_data['instances']
                           if inst.get('bbox_score', 0) != 1.0]

        if not valid_instances:
            return self._get_empty_features()

        # Process the most confident valid instance
        best_instance = self._get_best_instance(valid_instances)
        if not best_instance:
            return self._get_empty_features()

        keypoints = np.array(best_instance['keypoints'])
        scores = np.array(best_instance['keypoint_scores'])

        # Filter out low-confidence keypoints
        valid_mask = scores > 0.2
        if np.sum(valid_mask) < 10:  # Need at least 10 valid keypoints
            return self._get_empty_features()

        # Extract finger-specific features only
        features.update(self._extract_finger_features(keypoints, valid_mask))

        return features

    def _get_best_instance(self, instances: List[Dict]) -> Optional[Dict]:
        """Select the instance with highest bbox confidence."""
        if not instances:
            return None
        return max(instances, key=lambda x: x.get('bbox_score', 0))

    def _extract_finger_features(self, keypoints: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
        """Extract finger-specific features that distinguish eating vs drinking."""
        features = {}

        # For each finger, calculate extension and curl
        for finger_name, indices in self.keypoint_mapping.items():
            if finger_name == 'wrist':
                continue

            finger_features = self._analyze_finger(keypoints, valid_mask, indices, finger_name)
            features.update(finger_features)

        # Calculate overall finger metrics
        finger_extensions = [features.get(f'{name}_extension', 0)
                             for name in ['thumb', 'index', 'middle', 'ring', 'pinky']]
        features['fingers_extended_count'] = sum(1 for ext in finger_extensions if ext > 0.5)
        features['avg_finger_extension'] = float(np.mean(finger_extensions))

        # Pinch detection (thumb-index distance)
        if (len(keypoints) > 8 and valid_mask[4] and valid_mask[8]):  # Thumb tip and index tip
            thumb_tip = keypoints[4]
            index_tip = keypoints[8]
            pinch_distance = np.linalg.norm(thumb_tip - index_tip)
            features['pinch_distance'] = float(pinch_distance)
            features['is_pinching'] = float(pinch_distance < 30)  # Threshold for pinch

        return features

    def _analyze_finger(self, keypoints: np.ndarray, valid_mask: np.ndarray,
                        indices: List[int], finger_name: str) -> Dict[str, float]:
        """Analyze individual finger for extension/curl."""
        features = {}

        # Check if we have enough valid points for this finger
        valid_finger_points = [i for i in indices if i < len(valid_mask) and valid_mask[i]]

        if len(valid_finger_points) < 3:
            features[f'{finger_name}_extension'] = 0.0
            return features

        # Calculate finger extension by measuring the distance from base to tip
        # relative to the expected length
        base_idx = valid_finger_points[0]
        tip_idx = valid_finger_points[-1]

        base_point = keypoints[base_idx]
        tip_point = keypoints[tip_idx]

        # Distance from base to tip
        finger_length = np.linalg.norm(tip_point - base_point)

        # Calculate extension ratio (this is a simplified measure)
        # In a fully extended finger, the tip should be far from the base
        # For drinking, fingers are often more extended; for eating, more curled

        # Normalize by typical finger length (rough estimate)
        typical_finger_length = 80.0  # pixels, adjust based on your data
        extension_ratio = min(finger_length / typical_finger_length, 1.0)

        features[f'{finger_name}_extension'] = float(extension_ratio)

        return features

    def _get_empty_features(self) -> Dict[str, float]:
        """Return dictionary of empty/default features when no valid data."""
        return {
            'thumb_extension': 0.0, 'index_extension': 0.0, 'middle_extension': 0.0,
            'ring_extension': 0.0, 'pinky_extension': 0.0,
            'fingers_extended_count': 0.0, 'avg_finger_extension': 0.0,
            'pinch_distance': 0.0, 'is_pinching': 0.0
        }

    def process_video_frames(self, frames_data: List[Dict]) -> List[Dict[str, float]]:
        """
        Process multiple frames and return finger features for each.

        Args:
            frames_data: List of frame dictionaries

        Returns:
            List of feature dictionaries, one per frame
        """
        features_list = []

        for frame_data in frames_data:
            features = self.extract_features(frame_data)
            features['frame_id'] = frame_data.get('frame_id', 0)
            features_list.append(features)

        return features_list


    def create_feature_dataframe(self, video_paths: List[str], labels: List[str] = None) -> 'pd.DataFrame':
        """
        Create a pandas DataFrame from multiple video prediction files.

        Args:
            video_paths: List of paths to JSON files containing hand pose predictions
            labels: Optional list of labels corresponding to each video

        Returns:
            pandas DataFrame with features and labels, empty instances dropped
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for create_feature_dataframe. Install with: pip install pandas")

        all_features = []

        for i, video_path in enumerate(video_paths):
            try:
                with open(video_path, 'r') as f:
                    video_data = json.load(f)

                # Extract features for all frames in this video
                frame_features = self.process_video_frames(video_data)

                # Add video-level metadata
                for features in frame_features:
                    features['video_path'] = video_path
                    features['video_filename'] = video_path.split('/')[-1]
                    if labels:
                        features['label'] = labels[i]

                    all_features.append(features)

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(all_features)

        # Drop empty instances (where all finger features are 0)
        finger_features = ['thumb_extension', 'index_extension', 'middle_extension',
                           'ring_extension', 'pinky_extension']

        # Keep only rows where at least one finger feature is non-zero
        mask = (df[finger_features] != 0).any(axis=1)
        df_filtered = df[mask].copy()

        print(f"Original frames: {len(df)}")
        print(f"Frames after dropping empty instances: {len(df_filtered)}")
        print(f"Dropped {len(df) - len(df_filtered)} empty frames")

        return df_filtered


# Example usage and testing
def example_usage():
    """Example of how to use the feature generator."""

    # Sample data structure (based on your provided format)
    sample_data = [
        {
            "frame_id": 0,
            "instances": [
                {
                    "keypoints": [
                        [539.99, 416.46], [555.21, 386.51], [581.97, 357.54],
                        # ... more keypoints
                    ],
                    "keypoint_scores": [0.42, 0.55, 0.24, 0.20, 0.22],
                    "bbox": [[557.76, 303.59, 647.92, 404.14]],
                    "bbox_score": 0.35
                }
            ]
        }
    ]

    # Initialize feature generator
    feature_gen = HandPoseFeatureGenerator()

    # Extract features
    features_list = feature_gen.process_video_frames(sample_data)

    # Print some key features
    for i, features in enumerate(features_list):
        print(f"Frame {i} key features:")
        print(f"  Hand position: ({features['hand_center_x']:.1f}, {features['hand_center_y']:.1f})")
        print(f"  Hand height normalized: {features['hand_y_normalized']:.3f}")
        print(f"  Fingers extended: {features['fingers_extended_count']}")
        print(f"  Hand tilt: {features['hand_tilt']:.3f}")
        print(f"  Pinching: {features['is_pinching']}")
        print()


if __name__ == "__main__":
    example_usage()
