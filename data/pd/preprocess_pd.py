import argparse
import os
import pandas as pd
import numpy as np
import c3d
import csv

from const_pd import H36M_FULL, PD

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation


def convert_pd_h36m(sequence):
    new_keyponts = np.zeros((sequence.shape[0], 17, 3))
    new_keyponts[..., H36M_FULL['B.TORSO'], :] = (sequence[..., PD['L.ASIS'], :] +
                                             sequence[..., PD['R.ASIS'], :] +
                                             sequence[..., PD['L.PSIS'], :] +
                                             sequence[..., PD['R.PSIS'], :]) / 4
    new_keyponts[..., H36M_FULL['L.HIP'], :] = (sequence[..., PD['L.ASIS'], :] + 
                                           sequence[..., PD['L.PSIS'], :]) / 2
    new_keyponts[..., H36M_FULL['L.KNEE'], :] = sequence[..., PD['L.KNEE'], :]
    new_keyponts[..., H36M_FULL['L.FOOT'], :] = sequence[..., PD['L.ANKLE'], :]
    new_keyponts[..., H36M_FULL['R.HIP'], :] = (sequence[..., PD['R.ASIS'], :] + 
                                           sequence[..., PD['R.PSIS'], :]) / 2
    new_keyponts[..., H36M_FULL['R.KNEE'], :] = sequence[..., PD['R.KNEE'], :]
    new_keyponts[..., H36M_FULL['R.FOOT'], :] = sequence[..., PD['R.ANKLE'], :]
    new_keyponts[..., H36M_FULL['U.TORSO'], :] = (sequence[..., PD['C7'], :] + 
                                             sequence[..., PD['CLAV'], :]) / 2
    new_keyponts[..., H36M_FULL['C.TORSO'], :] = (sequence[..., PD['STRN'], :] + 
                                             sequence[..., PD['T10'], :]) / 2
    new_keyponts[..., H36M_FULL['R.SHOULDER'], :] = sequence[..., PD['R.SHO'], :]
    new_keyponts[..., H36M_FULL['R.ELBOW'], :] = (sequence[..., PD['R.EL'], :] + 
                                           sequence[..., PD['R.EM'], :]) / 2
    new_keyponts[..., H36M_FULL['R.HAND'], :] = (sequence[..., PD['R.WL'], :] + 
                                           sequence[..., PD['R.WM'], :]) / 2
    new_keyponts[..., H36M_FULL['L.SHOULDER'], :] = sequence[..., PD['L.SHO'], :]
    new_keyponts[..., H36M_FULL['L.ELBOW'], :] = (sequence[..., PD['L.EL'], :] + 
                                           sequence[..., PD['L.EM'], :]) / 2
    new_keyponts[..., H36M_FULL['L.HAND'], :] = (sequence[..., PD['L.WL'], :] + 
                                           sequence[..., PD['L.WM'], :]) / 2
    new_keyponts[..., H36M_FULL['NECK'], :] = new_keyponts[..., H36M_FULL['U.TORSO'], :] + [0.27, 57.48, 11.44]
    new_keyponts[..., H36M_FULL['HEAD'], :] = new_keyponts[..., H36M_FULL['U.TORSO'], :] + [-2.07, 165.23, 34.02]
    
    return new_keyponts

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='/mnt/Ndrive/AMBIENT/Vida/Public_datasets/pd/14896881/', type=str, help='Path to the input folder')
    args = parser.parse_args()
    return args


def rotate_around_z_axis(points, theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points, R.T)


def visualize_sequence(seq, name):
    VIEWS = {
    "pd": {
        "best": (45, 20, 100),
        "best2": (0, 0, 0),
        "side": (90, 0, 90),
    },
    "tmp": {
        "best": (45, 20, 100),
        "side": (90, 0, 90),
    }
}
    elev, azim, roll = VIEWS["pd"]["best"]
    # Apply the rotation to each point in the sequence
    for i in range(seq.shape[1]):
        seq[:, i, :] = rotate_around_z_axis(seq[:, i, :], roll)

    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])

        # print(VIEWS[data_type][view_type])
        # ax.view_init(*VIEWS[data_type][view_type])
        elev, azim, roll = VIEWS["pd"]["best"]
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame}')

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        # for connection in connections:
        #     start = seq[frame, connection[0], :]
        #     end = seq[frame, connection[1], :]
        #     xs = [start[0], end[0]]
        #     ys = [start[1], end[1]]
        #     zs = [start[2], end[2]]

        #     ax.plot(xs, ys, zs)
        ax.scatter(x, y, z)

    
    print(f"Number of frames: {seq.shape[0]}")

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create the animation
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    ani.save(f'{name}.gif', writer='pillow')
    
    plt.close(fig)

def read_pd(sequence_path, start_index, step):
    """
    Read points data from a .c3d file and create a sequence of selected frames.

    Parameters:
    sequence_path (str): The file path for the .c3d file.
    start_index (int): The frame index at which to start reading the data.
    step (int): The number of frames to skip between reads. A step of n reads every nth frame.

    Returns:
    numpy.ndarray: An array containing the processed sequence of points data from the .c3d file.

    """
    reader = c3d.Reader(open(sequence_path, 'rb'))
    sequence = []
    for i, points, analog in reader.read_frames():
        if i >= start_index and (i - start_index) % step == 0:
            if np.any(np.all(points[:44, :3] == 0, axis=1)):   #Removed frames with corrupted joints
                continue
            sequence.append(points[None, :44, :3])
    if len(sequence) == 0:
        print(sequence_path)
        with open('./data/pd/Removed_sequences.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sequence_path])
        return sequence
        # sequence2 = []
        # for i, points, analog in reader.read_frames():
        #     if i >= start_index and (i - start_index) % step == 0:
        #         sequence2.append(points[None, :44, :3])
        # sequence2 = np.concatenate(sequence2)
        # visualize_sequence(sequence2, './data/pd/orig_allremoved')
    sequence = np.concatenate(sequence)

    sequence = convert_pd_h36m(sequence)
    # visualize_sequence(sequence, './data/pd/orig_all')
    return sequence

def main():
    args = parse_args()

    input_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles')
    output_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles_processed_new')

    if not os.path.exists(input_path_c3dfiles):
        raise FileNotFoundError(f"Input folder '{input_path_c3dfiles}' not found.")

    os.makedirs(output_path_c3dfiles, exist_ok=True)
    for root, dirs, files in os.walk(input_path_c3dfiles):
        for file in files:
            if file.endswith('.c3d') and "walk" in file and file.startswith("SUB"):
                sequence_path = os.path.join(root, file)
                try:
                    for start_index in range(3):
                        sequence = read_pd(sequence_path, start_index, 3)
                        if len(sequence) == 0:
                            continue
                        output_sequence_path = os.path.join(output_path_c3dfiles, f"{file[:-4]}_{start_index}")
                        print(output_sequence_path)
                        np.save(output_sequence_path + '.npy', sequence)
                except Exception as e:
                    print(f"Error reading {sequence_path}: {str(e)}")

if __name__ == "__main__":
    main()