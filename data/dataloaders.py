import torch.utils.data as data
from abc import ABC
from torchvision import transforms
import torch
import os

import pickle
import random
import copy
import pandas as pd

from const import path
from data.utility import *
from data.data_augmentation import *
from data.public_pd_datareader import PDReader
from data.augmentations import MirrorReflection, RandomRotation, RandomNoise, axis_mask
from learning.utils import compute_class_weights

_TOTAL_SCORES = 3
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#                1,   2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21
_ROOT = 0
_MIN_STD = 1e-4

METADATA_MAP = {'gender': 0, 'age': 1, 'height': 2, 'weight': 3, 'bmi': 4}


class DataPreprocessor(ABC):
    def __init__(self, raw_data, params=None):
        self.pose_dict = raw_data.pose_dict
        self.labels_dict = raw_data.labels_dict
        self.metadata_dict = raw_data.metadata_dict
        self.video_names = raw_data.video_names
        self.participant_ID = raw_data.participant_ID
        self.params = params

        self.data_dir = self.params['data_path']

    def __len__(self):
        return len(self.labels_dict)

    def center_poses(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            self.pose_dict[key] = joints3d - joints3d[:, _ROOT:_ROOT + 1, :]

    def normalize_poses(self):
        if self.params['data_norm'] == 'minmax':
            """
            Normalize each pose along each axis by video. Divide by the largest value in each direction
            and center around the origin.
            :param pose_dict: dictionary of poses
            :return: dictionary of normalized poses
            """
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(np.min(poses, axis=0), axis=0)
                maxes = np.max(np.max(poses, axis=0), axis=0)

                poses = (poses - mins) / (maxes - mins)

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'rescaling':
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(poses, axis=(0, 1))
                maxes = np.max(poses, axis=(0, 1))

                poses = (2 * (poses - mins) / (maxes - mins)) - 1

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'zscore':
            norm_stats = self.compute_norm_stats()
            pose_dict_norm = self.pose_dict.copy()
            for k in self.pose_dict.keys():
                tmp_data = self.pose_dict[k].copy()
                tmp_data = tmp_data - norm_stats['mean']
                tmp_data = np.divide(tmp_data, norm_stats['std'])
                pose_dict_norm[k] = tmp_data
            self.pose_dict = pose_dict_norm

    def compute_norm_stats(self):
        all_data = []
        for k in self.pose_dict.keys():
            all_data.append(self.pose_dict[k])
        all_data = np.vstack(all_data)
        print('[INFO] ({}) Computing normalization stats!')
        norm_stats = {}
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        std[np.where(std < _MIN_STD)] = 1

        norm_stats['mean'] = mean  # .ravel()
        norm_stats['std'] = std  # .ravel()
        return norm_stats
        
    def generate_leave_one_out_folds(self, clip_dict, save_dir, labels_dict):
        """
        Generate folds for leave-one-out CV.
        :param clip_dict: dictionary of clips for each video
        :param save_dir: save directory for folds
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        video_names_list = list(clip_dict.keys())
        fold = 0

        dataset_name = self.params['dataset']
        val_folds_name = 'val_PD_SUBs_folds.pkl' if dataset_name == 'PD' else 'val_AMBIDs_folds.pkl'
        val_folds_path = os.path.join(save_dir, '..', val_folds_name)

        val_folds_exists = os.path.exists(val_folds_path)

        if not val_folds_exists:
            val_subs_folds = []
            print(f'[INFO] Previous selected {dataset_name} validation set does not exist.')
        else:
            val_subs_folds = pickle.load(open(val_folds_path, "rb"))


        for j in range(len(self.participant_ID)):
            train_list, val_list, test_list = [], [], []

            participant_ID_cloned = copy.deepcopy(self.participant_ID)
            subject_id = participant_ID_cloned.pop(j)

            class_participants = {}
            for participant in participant_ID_cloned:
                participant_labels = [labels_dict[key] for key in labels_dict if key.startswith(participant + "_on") or key.startswith(participant + "_off")]
                if participant_labels:
                    class_participants[participant] = participant_labels[0]  # Use the first label as the class

            if not val_folds_exists:
                val_subs = []
                for class_id in range(3):  # Assuming classes are 0, 1, 2
                    class_participants_for_class = [participant for participant, class_label in class_participants.items() if class_label == class_id]
                    for _ in range(2):  # Select 2 participants from each class
                        val_idx = random.randint(0, len(class_participants_for_class) - 1)
                        val_subs.append(class_participants_for_class.pop(val_idx))
                val_subs_folds.append(val_subs)
            else:
                val_subs = val_subs_folds[j]

            for k in range(len(video_names_list)):
                video_name = video_names_list[k]
                # augmented = any(augmentation in video_name for augmentation in self.params['augmentation'])
                if dataset_name == 'PD':
                    if subject_id == video_name.split("_")[0]:
                        # if not augmented:
                        test_list.append(video_name)
                    elif video_name.split("_")[0] in val_subs:
                        val_list.append(video_name)
                    else:
                        train_list.append(video_name)
            print("Fold: ", fold)
            fold += 1
            train, validation, test = self.generate_pose_label_videoname(clip_dict, train_list, val_list, test_list)
            pickle.dump(train_list, open(os.path.join(save_dir, f"{dataset_name}_train_list_{fold}.pkl"), "wb"))
            pickle.dump(test_list, open(os.path.join(save_dir, f"{dataset_name}_test_list_{fold}.pkl"), "wb"))
            pickle.dump(val_list, open(os.path.join(save_dir, f"{dataset_name}_validation_list_{fold}.pkl"), "wb"))
            pickle.dump(train, open(os.path.join(save_dir, f"{dataset_name}_train_{fold}.pkl"), "wb"))
            pickle.dump(test, open(os.path.join(save_dir, f"{dataset_name}_test_{fold}.pkl"), "wb"))
            pickle.dump(validation, open(os.path.join(save_dir, f"{dataset_name}_validation_{fold}.pkl"), "wb"))
        pickle.dump(self.labels_dict, open(os.path.join(save_dir, f"{dataset_name}_labels.pkl"), "wb"))

        if not val_folds_exists:
            pickle.dump(val_subs_folds, open(val_folds_path, "wb"))


    def get_data_split(self, split_list, clip_dict):
        split = {'pose': [], 'label': [], 'video_name': [], 'metadata': []}
        for video_name in split_list:
            clips = clip_dict[video_name]
            for clip in clips:
                split['label'].append(self.labels_dict[video_name])
                split['pose'].append(clip)
                split['video_name'].append(video_name)
                split['metadata'].append(self.metadata_dict[video_name])
        return split

    def generate_pose_label_videoname(self, clip_dict, train_list, val_list, test_list):
        train = self.get_data_split(train_list, clip_dict)
        val = self.get_data_split(val_list, clip_dict)
        test = self.get_data_split(test_list, clip_dict)

        #print how many samples are in each split
        print(f"Train Length: {len(train['video_name'])}")
        print(f"Validation Length: {len(val['video_name'])}")
        print(f"Test Length: {len(test['video_name'])}")
        return train, val, test

    @staticmethod
    def resample(original_length, target_length):
        """
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68

        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result


class POTRPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)
        self.length = self.params['source_seq_len']

        if self.params['data_centered']:
            self.center_poses()
        if self.params['data_norm'] in ['minmax', 'zscore']:
            # ToDo: check
            self.normalize_poses()
        clip_dict = self.partition_videos()
        self.generate_leave_one_out_folds(clip_dict, save_dir, raw_data.labels_dict)

    # def partition_videos(self, offset=50):
    #     """
    #     Partition poses from each video into clips.
    #     :param offset: offset between clips
    #     :return: dictionary of clips for each video
    #     """
    #     clip_dict = {}
    #     seqs_len = []
    #     for name in self.video_names:
    #         seqs_len.append(len(self.pose_dict[name]))
    #         if not self.params['interpolate']:
    #             if len(self.pose_dict[name]) < self.length:
    #                 continue
    #         clips = self.get_clips(self.pose_dict[name], self.length)
    #         # clips = get_clips_overlap(pose_dict[name], length, offset)
    #         clip_dict[name] = clips
    #     return clip_dict

    # def get_clips(self, video, clip_length):
    #     """
    #     Returns a list of partitioned gait segments of given length in frames
    #     :param video: input video
    #     :param length: length of clip
    #     :return: partition of gait segments into clips
    #     """
    #     clips = []
    #     video_length = len(video)
    #     # TODo: this should be changed so that we won't have different random start for each fold
    #     if video_length <= clip_length:
    #         new_indices = self.resample(video_length, clip_length)
    #         clips.append(video[new_indices])
    #     else:
    #         start = random.randint(0, video_length - clip_length)
    #         end = start + clip_length
    #         clip = np.array(video[start:end])
    #         clips.append(clip)

    #     return clips
    def partition_videos(self):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], self.length)
            clip_dict[video_name] = clips
        return clip_dict
    
    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pass
            #new_indices = self.resample(video_length, clip_length)
            #clips.append(video_sequence[new_indices])
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips


class MotionBERTPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()
        else:
            self.place_depth_of_first_frame_to_zero()

        clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_leave_one_out_folds(clip_dict, save_dir, raw_data.labels_dict)

    def place_depth_of_first_frame_to_zero(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints3d[..., 2] = joints3d[..., 2] - joints3d[0:1, _ROOT:_ROOT + 1, 2]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
        return clip_dict

    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pass
            #new_indices = self.resample(video_length, clip_length)
            #clips.append(video_sequence[new_indices])
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips
    

class MotionAGFormerPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()
        else:
            self.place_depth_of_first_frame_to_zero()

        clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_leave_one_out_folds(clip_dict, save_dir)

    def place_depth_of_first_frame_to_zero(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints3d[..., 2] = joints3d[..., 2] - joints3d[0:1, _ROOT:_ROOT + 1, 2]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
        return clip_dict

    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pass
            #new_indices = self.resample(video_length, clip_length)
            #clips.append(video_sequence[new_indices])
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips


class PoseformerV2Preprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()

        self.remove_last_dim_of_pose()
        self.normalize_poses()
        clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_leave_one_out_folds(clip_dict, save_dir)

    def remove_last_dim_of_pose(self):
        for video_name in self.pose_dict:
            self.pose_dict[video_name] = self.pose_dict[video_name][..., :2]  # Ignoring confidence score

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
        return clip_dict

    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pass
            #new_indices = self.resample(video_length, clip_length)
            #clips.append(video_sequence[new_indices])
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips

class MixSTEPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()

        self.remove_last_dim_of_pose()
        self.normalize_poses()
        clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_leave_one_out_folds(clip_dict, save_dir)

    def remove_last_dim_of_pose(self):
        for video_name in self.pose_dict:
            self.pose_dict[video_name] = self.pose_dict[video_name][..., :2]  # Ignoring confidence score

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
        return clip_dict

    def get_clips(self, video_sequence, clip_length):
        clips = []
        video_length = video_sequence.shape[0]
        if video_length <= clip_length:
            new_indices = self.resample(video_length, clip_length)
            clips.append(video_sequence[new_indices])
        else:
            middle_frame = (video_length) // 2
            start_frame = middle_frame - (clip_length // 2)
            clip = video_sequence[start_frame: start_frame + clip_length]
            clips.append(clip)
        return clips

class ProcessedDataset(data.Dataset):
    def __init__(self, data_dir, params=None, mode='train', fold=1, downstream='pd', transform=None):
        super(ProcessedDataset, self).__init__()
        self._params = params
        self._mode = mode
        self.data_dir = data_dir
        self._task = downstream
        self._NMAJOR_JOINTS = len(_MAJOR_JOINTS)

        

        if self._task == 'pd':
            self._updrs_str = ['normal', 'slight', 'moderate']  # , 'severe']
            self._TOTAL_SCORES = _TOTAL_SCORES

        self.fold = fold
        self.transform = transform

        self.poses, self.labels, self.video_names, self.metadata = self.load_data()
        self.video_name_to_index = {name: index for index, name in enumerate(self.video_names)}


        self._pose_dim = 3 * self._NMAJOR_JOINTS

    def load_data(self):
        dataset_name = self._params['dataset']

        if self._mode == 'train':
            train_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_train_{self.fold}.pkl"), "rb"))

        elif self._mode == 'test':
            test_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_test_{self.fold if self._mode != 'test_all' else 'all'}.pkl"), "rb"))

        elif self._mode == 'val':
            val_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_validation_{self.fold}.pkl"), "rb"))
        elif self._mode == 'train-eval':
            train_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_train_{self.fold}.pkl"), "rb"))
            val_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_validation_{self.fold}.pkl"), "rb"))
            train_data = {
                'pose': [*train_data['pose'], *val_data['pose'],],
                'label': [*train_data['label'], *val_data['label']],
                'video_name': [*train_data['video_name'], *val_data['video_name']]
            }
        elif self._mode == 'test_all':
            test_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_all.pkl"), "rb"))


        if self._mode == 'train':
            poses, labels, video_names, metadatas = self.data_generator(train_data, mode='train', fold_number=self.fold)
        elif self._mode == 'test':
            poses, labels, video_names, metadatas = self.data_generator(test_data)
        elif self._mode == 'val':
            poses, labels, video_names, metadatas = self.data_generator(val_data)
        elif self._mode == 'train-eval':
            poses, labels, video_names, metadatas = self.data_generator(train_data, mode='train', fold_number=self.fold)
        else:
            poses, labels, video_names, metadatas = self.data_generator(test_data)

        return poses, labels,video_names, metadatas

    @staticmethod
    def data_generator(data, mode='test', fold_number=1):
        poses = []
        labels = []
        video_names = []
        metadatas = []

        # bootstrap_number = 3
        # num_samples = 39

        for i in range(len(data['pose'])):
            pose = np.copy(data['pose'][i])
            label = data['label'][i]
            poses.append(pose)
            labels.append(label)
            video_names.append(data['video_name'][i])
            metadata = data['metadata'][i]
            metadatas.append(metadata)
        # can't stack poses because not all have equal frames
        labels = np.stack(labels)
        video_names = np.stack(video_names)
        metadatas = np.stack(metadatas)

        # For using a subset of the dataset (few-shot)
        # if mode == 'train':
        #   sampling_dir = 'PATH/TO/BOOTSTRAP_SAMPLING_DIR'
        #   all_clip_video_names = pickle.load(open(sampling_dir + "all_clip_video_names.pkl", "rb"))
        #   clip_video_names = all_clip_video_names[fold_number - 1]

        #   all_bootstrap_samples = pickle.load(open(sampling_dir + f'{num_samples}_samples/bootstrap_{bootstrap_number}_samples.pkl', 'rb'))
        #   bootstrap_samples = all_bootstrap_samples[fold_number - 1]

        #   mask_list = [1 if video_name in bootstrap_samples else 0 for video_name in clip_video_names]
        #   train_indices = [train_idx for train_idx, mask_value in enumerate(mask_list) if mask_value == 1]

        #   X_1 = [X_1[train_idx] for train_idx in train_indices]
        #   Y = Y[train_indices]

        return poses, labels, video_names, metadatas

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for the training mode."""
        x = self.poses[idx]
        label = self.labels[idx]
        video_idx = self.video_name_to_index[self.video_names[idx]] 

        if self._params['data_type'] != "GastNet":
            joints = self._get_joint_orders()
            x = x[:, joints, :]

        if self._params['in_data_dim'] == 2:
            if self._params['simulate_confidence_score']:
                # TODO: Confidence score should be a function of depth (probably)
                x[..., 2] = 1  # Consider 3rd dimension as confidence score and set to be 1.
            else:
                x = x[..., :2]  # Make sure it's two-dimensional
        elif self._params['in_data_dim'] == 3:
            x = x[..., :3] # Make sure it's 3-dimensional
                

        if self._params['merge_last_dim']:
            N = np.shape(x)[0]
            x = x.reshape(N, -1)   # N x 17 x 3 -> N x 51

        x = np.array(x, dtype=np.float32)

        if x.shape[0] > self._params['source_seq_len']:
            # If we're reading a preprocessed pickle file that has more frames
            # than the expected frame length, we throw away the last few ones.
            x = x[:self._params['source_seq_len']]
        elif x.shape[0] < self._params['source_seq_len']:
            raise ValueError("Number of frames in tensor x is shorter than expected one.")
        
        if len(self._params['metadata']) > 0:
            metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
            md = self.metadata[idx][0][metadata_idx].astype(np.float32)
        else:
            md = []

        sample = {
            'encoder_inputs': x,
            'label': label,
            'labels_str': self._updrs_str[label],
            'video_idx': video_idx,
            'metadata': md,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_joint_orders(self):
        joints = _MAJOR_JOINTS
        return joints

def collate_fn(batch):


    """Collate function for data loaders."""
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    labels = torch.from_numpy(np.stack([e['label'] for e in batch]))
    video_idxs = torch.from_numpy(np.stack([e['video_idx'] for e in batch]))
    metadata = torch.from_numpy(np.stack([e['metadata'] for e in batch]))

    # action = [e['labels_str'] for e in batch]

    # batch_ = {
    #     'encoder_inputs': e_inp,
    #     'labels_str': action,
    #     'labels': label
    # }
    # return batch_
    return e_inp, labels, video_idxs, metadata


def assert_backbone_is_supported(backbone_data_location_mapper, backbone):
    if backbone not in backbone_data_location_mapper:
        raise NotImplementedError(f"Backbone '{backbone}' is not supported.")

def dataset_factory(params, backbone, fold):
    """Defines the datasets that will be used for training and validation."""
    params['n_joints'] = len(_MAJOR_JOINTS)

    root_dir = f'{path.PREPROCESSED_DATA_ROOT_PATH}/{backbone}_processing'

    backbone_data_location_mapper = {
        'poseformer': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}_{params['data_norm']}/"),
        'motionbert': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}/"),
        'motionagformer': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}/"),
        'poseformerv2': os.path.join(root_dir, params['experiment_name'],
                                     f"{params['dataset']}_center_{params['data_centered']}/"),
        'mixste': os.path.join(root_dir, params['experiment_name'],
                                     f"{params['dataset']}_center_{params['data_centered']}/"),
    }

    backbone_preprocessor_mapper = {
        'poseformer': POTRPreprocessor,
        'motionbert': MotionBERTPreprocessor,
        'poseformerv2': PoseformerV2Preprocessor,
        'mixste': MixSTEPreprocessor,
        'motionagformer': MotionAGFormerPreprocessor
    }

    assert_backbone_is_supported(backbone_data_location_mapper, backbone)
    
    data_dir = backbone_data_location_mapper[backbone]

    if not os.path.exists(data_dir):
        if params['dataset'] == 'PD':
            raw_data = PDReader(params['data_path'], params['labels_path']) 
            
            # if params['augmentation']:
            #     augmenter = PoseSequenceAugmentation(params)
            #     raw_data = augmenter.augment_data(raw_data, params['augmentation'])

            Preprocessor = backbone_preprocessor_mapper[backbone]
            Preprocessor(data_dir, raw_data, params)
        else:
            raise NotImplementedError(f"dataset '{params['dataset']}' is not supported.")

    use_validation = params['use_validation']

    train_transform = transforms.Compose([
        PreserveKeysTransform(transforms.RandomApply([MirrorReflection(data_dim=params['in_data_dim'])], p=params['mirror_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomRotation(*params['rotation_range'], data_dim=params['in_data_dim'])], p=params['rotation_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomNoise(data_dim=params['in_data_dim'])], p=params['noise_prob'])),
        PreserveKeysTransform(transforms.RandomApply([axis_mask(data_dim=params['in_data_dim'])], p=params['axis_mask_prob']))
    ])

    train_dataset = ProcessedDataset(data_dir, fold=fold, params=params,
                                            mode='train' if use_validation else 'train-eval', transform=train_transform)
    eval_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='val') if use_validation else None
    test_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='test')
    
    #Uncomment to save information about each subset of Data (SUB IDs and number of scores within each set)
    # csv_file = 'folds_subs.csv'
    # row = {
    #     'foldnumber': fold,
    #     'TrainSUB': ', '.join(extract_unique_subs(train_dataset)),
    #     'ValSUB': ', '.join(extract_unique_subs(eval_dataset)),
    #     'TestSUB': ', '.join(extract_unique_subs(test_dataset))
    # }
    # all_labels = set(range(0, train_dataset._params['num_classes']))
    # train_label_counts = count_labels(train_dataset, all_labels)
    # val_label_counts = count_labels(eval_dataset, all_labels)
    # test_label_counts = count_labels(test_dataset, all_labels)
    # for lbl, count in train_label_counts.items():
    #     row[f'Train_score{lbl}'] = count
    # for lbl, count in val_label_counts.items():
    #     row[f'Val_score{lbl}'] = count
    # for lbl, count in test_label_counts.items():
    #     row[f'Test_score{lbl}'] = count
    # df = pd.DataFrame([row])
    # header = not pd.io.common.file_exists(csv_file)
    # df.to_csv(csv_file, mode='a', index=False, header=header)

    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    eval_dataset_fn = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    ) if use_validation else None

    test_dataset_fn = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    class_weights = compute_class_weights(train_dataset_fn)

    return train_dataset_fn, test_dataset_fn, eval_dataset_fn, class_weights

class PreserveKeysTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        transformed_sample = self.transform(sample)

        # Ensure all original keys are preserved
        for key in sample.keys():
            if key not in transformed_sample:
                transformed_sample[key] = sample[key]

        return transformed_sample
