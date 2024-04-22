import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *

class PDReader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    ON_LABEL_COLUMN = 'ON - UPDRS-III - walking'
    OFF_LABEL_COLUMN = 'OFF - UPDRS-III - walking'
    DELIMITER = ';'

    def __init__(self, joints_path, labels_path):
        self.joints_path = joints_path
        self.labels_path = labels_path
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints_and_labels()

    def read_sequence(self, path_file):
        """
        Reads skeletons from npy files
        """
        if os.path.exists(path_file):
            body = np.load(path_file)
            body = body/1000 #convert mm to m
        else:
            body = None
        return body

    def read_label(self, file_name):
        subject_id, on_or_off = file_name.split("_")[:2]
        df = pd.read_excel(self.labels_path)
        # df = pd.read_csv(self.labels_path, delimiter=self.DELIMITER)
        df = df[['ID', self.ON_LABEL_COLUMN, self.OFF_LABEL_COLUMN]]
        subject_rows = df[df['ID'] == subject_id]
        if on_or_off == "on":
            label = subject_rows[self.ON_LABEL_COLUMN].values[0]
        else:
            label = subject_rows[self.OFF_LABEL_COLUMN].values[0]
        return int(label)
    
    def read_metadata(self, file_name):
        #If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        subject_id = file_name.split("_")[0]
        df = pd.read_excel(self.labels_path)
        df = df[['ID', 'Gender', 'Age', 'Height (cm)', 'Weight (kg)', 'BMI (kg/m2)']]
        df.rename(columns={
            "Gender": "gender",
            "Age": "age",
            "Height (cm)": "height",
            "Weight (kg)": "weight",
            "BMI (kg/m2)": "bmi"}, inplace=True)
        df.loc[:, 'gender'] = df['gender'].map({'M': 0, 'F': 1})
        
        # Using Min-Max normalization
        df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        df['height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
        df['weight'] = (df['weight'] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())
        df['bmi'] = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())

        subject_rows = df[df['ID'] == subject_id]
        return subject_rows.values[:, 1:] 
    
    def read_keypoints_and_labels(self):
        """
        Read npy files in given directory into arrays of pose keypoints.
        :return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        labels_dict = {}
        metadata_dict = {}
        video_names_list = []
        participant_ID = []

        print('[INFO - PublicPDReader] Reading body keypoints from npy')

        print(self.joints_path)

        for file_name in tqdm(os.listdir(self.joints_path)):
            path_file = os.path.join(self.joints_path, file_name)
            joints = self.read_sequence(path_file)
            label = self.read_label(file_name)
            metadata = self.read_metadata(file_name)
            if joints is None:
                print(f"[WARN - PublicPDReader] Numpy file {file_name} does not exist")
                continue
            file_name = file_name.split(".")[0]
            pose_dict[file_name] = joints
            labels_dict[file_name] = label
            metadata_dict[file_name] = metadata
            video_names_list.append(file_name)
            participant_ID.append(file_name.split("_")[0])

        participant_ID = self.select_unique_entries(participant_ID)

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict

    @staticmethod
    def select_unique_entries(a_list):
        return sorted(list(set(a_list)))

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for the training mode."""

        # Based on index, get the video name
        video_name = self.video_names[idx]

        x = self.poses[video_name]
        label = self.labels[video_name]
        

        x = np.array(x, dtype=np.float32)

        sample = {
            'encoder_inputs': x,
            'label': label,
            
        }
        #if self.transform:
        #    sample = self.transform(sample)

        return sample

# raw_data = PDReader('/data/iballester/datasets/Public_PD/C3Dfiles_processed/', '/data/iballester/datasets/Public_PD/PDGinfo.csv') 