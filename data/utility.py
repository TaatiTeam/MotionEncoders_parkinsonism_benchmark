import numpy as np

def walkid_to_AMBID(cur_walk_id):
    # Extract AMBID from the walk
    raw_id = cur_walk_id
    if raw_id >= 60:
        id = raw_id - 3
    else:
        id = raw_id - 2
    return id


def get_AMBID_from_Videoname(path_file):
    # The pattern is YYYY_MM_DD_hh_mm_ss_ID_XX_state_X.csv
    AMBID = walkid_to_AMBID(int(path_file[24:26]))
    AMBID = 'AMB' + str(AMBID).zfill(2)

    return AMBID


def extract_unique_subs(dataset):
    if dataset is None:
        return []
    unique_subs = set()
    for name in dataset.video_names:
        sub = name.split('_')[0]  # Assuming SUBXX is always the first part of the video name
        unique_subs.add(sub)
    return list(unique_subs)

def count_labels(dataset, all_labels):
    label_counts = {lbl: 0 for lbl in all_labels}  # Initialize all labels with count 0
    if dataset is not None:
        labels, counts = np.unique(dataset.labels, return_counts=True)
        label_counts.update(dict(zip(labels, counts)))
    return label_counts