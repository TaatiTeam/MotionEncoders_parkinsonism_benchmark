import copy

import numpy as np
from data.visualize import visualize_sequence


class PoseSequenceAugmentation:
    def __init__(self, params):
        self.augmentation_methods = {
            "mirror_reflection": self.mirror_reflection,
            "joint_dropout": self.joint_dropout,
            "random_rotation": self.random_rotation,
            "random_translation": self.random_translation
        }
        self.params = params

    def augment_data(self, raw_data, augmentation_list, visualize_only=False):
        if "random_translation" in augmentation_list:
            self.estimate_translation_range(raw_data.pose_dict)

        augmented_data_dict = {"pose_dict": {}, "labels_dict": {}}

        augmented_video_names = []
        for video_name, pose_sequence in raw_data.pose_dict.items():
            augmented_sequences = {}

            for augmentation_name in augmentation_list:
                if augmentation_name in self.augmentation_methods:
                    augmented_sequence = self.augmentation_methods[augmentation_name](pose_sequence)
                    augmented_sequences[augmentation_name] = augmented_sequence
                    if visualize_only:
                        visualize_sequence(pose_sequence, augmented_sequence, video_name + '_org')
                else:
                    print(f"Warning: Unknown augmentation technique '{augmentation_name}'")

            if visualize_only:
                exit()

            for augmentation_name, augmented_sequence in augmented_sequences.items():
                augmented_video_name = f"{video_name}_{augmentation_name}"
                augmented_video_names.append(augmentation_name)

                augmented_data_dict["pose_dict"][augmented_video_name] = augmented_sequence
                augmented_data_dict["labels_dict"][augmented_video_name] = raw_data.labels_dict[
                    video_name]

        return self.update_datareader(raw_data, augmented_data_dict, augmented_video_names)

    @staticmethod
    def update_datareader(raw_data, augmented_data_dict, augmented_video_names):
        raw_data_augmented = copy.deepcopy(raw_data)
        raw_data_augmented.labels = raw_data_augmented.labels + list(
            augmented_data_dict['labels_dict'].values())  # TODO: Should we also remove this?
        raw_data_augmented.video_names = raw_data_augmented.video_names + augmented_video_names
        raw_data_augmented.labels_dict.update(augmented_data_dict['labels_dict'])
        raw_data_augmented.pose_dict.update(augmented_data_dict['pose_dict'])
        return raw_data_augmented

    @staticmethod
    def mirror_reflection(pose_sequence):
        mirrored_sequence = pose_sequence.copy()
        left = [4, 5, 6, 10, 11, 12]
        right = [7, 8, 9, 13, 14, 15]
        mirrored_sequence[:, :, 0] *= -1
        mirrored_sequence[:, left + right, :] = mirrored_sequence[:, right + left, :]
        return mirrored_sequence

    @staticmethod
    def joint_dropout(pose_sequence, dropout_prob):
        # Randomly remove certain joints from the pose sequence
        dropout_mask = np.random.choice([0, 1], size=pose_sequence.shape[1], p=[dropout_prob, 1 - dropout_prob])
        dropped_sequence = pose_sequence * dropout_mask
        return dropped_sequence

    def random_rotation(self, pose_sequence):
        # Randomly rotate the pose sequence
        rotation_angles = np.random.uniform(self.params['rotation_range'][0], self.params['rotation_range'][1], size=3)
        rotation_matrix = self.rotation_matrix(rotation_angles)
        rotated_sequence = np.matmul(pose_sequence, rotation_matrix)
        return rotated_sequence

    def random_translation(self, pose_sequence):
        noise_scale = 0
        # # Randomly translate the pose sequence with added noise
        translation = np.random.uniform(self.translation_range[0], self.translation_range[1], size=3)
        noise = np.random.normal(scale=noise_scale, size=pose_sequence.shape)
        translated_sequence = pose_sequence + translation + noise
        return translated_sequence

    def estimate_translation_range(self, pose_dict):
        min_values = np.min([np.min(pose) for pose in pose_dict.values()])
        max_values = np.max([np.max(pose) for pose in pose_dict.values()])
        overall_range = max_values - min_values
        self.translation_range = (-self.params['translation_frac'] * overall_range,
                                  self.params['translation_frac'] * overall_range)  # Adjust the fraction (0.1)

    def estimate_noise_scale(pose_dict):
        min_values = np.min([np.min(pose) for pose in pose_dict.values()])
        max_values = np.max([np.max(pose) for pose in pose_dict.values()])
        overall_range = max_values - min_values
        noise_scale = 0.1 * overall_range  # Adjust the fraction (0.1) as desired
        return noise_scale

    @staticmethod
    def rotation_matrix(angles):
        radians = angles * (np.pi / 180)
        # Helper function to generate a rotation matrix
        alpha, beta, gamma = radians
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))
        return rotation_matrix
