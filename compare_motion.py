import os
import argparse
import copy


from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt  
from fastdtw import fastdtw
import numpy as np
import imageio

from lib.utils.utils_skeleton import flip_h36m_motion

from vispy import app, scene, io
from vispy.color import Color



JOINT_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
                            [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], 
                            [11, 12], [12, 13], [14, 15], [15, 16]])
JOINT_PAIRS_LEFT = np.array([[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]])
JOINT_PAIRS_RIGHT = np.array([[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]])

COLOR_MID = Color("#00457E").rgb
COLOR_LEFT = Color("#02315E").rgb
COLOR_RIGHT = Color("#2F70AF").rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_motion", type=str, required=True, help=".npy file of the student motion")
    parser.add_argument('--teacher_motion', type=str, required=True, help=".npy file of the teacher motion")
    opts = parser.parse_args()
    return opts

def load_skeleton(path): 
    skeleton_data = np.load(path) 
    return skeleton_data


def skeleton_distance(frame1, frame2):
    """Compute distance between two skeleton frames."""
    return np.sum(np.sqrt(np.sum((frame1 - frame2)**2, axis=1)))

def dtw_skeleton(seq1, seq2, plot=True):
    """
    Compute DTW for two skeleton sequences using fastdtw and optionally plot the result.
    
    :param seq1: NumPy array of shape (N, 17, 3)
    :param seq2: NumPy array of shape (M, 17, 3)
    :param plot: Boolean, whether to plot the result
    :return: DTW distance, warping path
    """
    distance, path = fastdtw(seq1, seq2, dist=skeleton_distance)
    
    plt.plot([x[0] for x in path], [x[1] for x in path])
    
    # Set labels and title
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.title(f'DTW Warping Path (Distance: {distance:2f})')
    
    # Set axis limits
    plt.xlim(0, len(seq1) - 1)
    plt.ylim(0, len(seq2) - 1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure the aspect ratio is equal
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig("plot.png")
    
    return distance, path


# @torch.no_grad()
# def _calculate_similarity(tch_kpts: np.ndarray, stu_kpts: np.ndarray):

#     if torch.cuda.is_available():
#         device = "cuda"
#     else: 
#         device = "cpu"

#     stu_kpts = torch.from_numpy(stu_kpts[:, None, :]).to(device)
#     tch_kpts = torch.from_numpy(tch_kpts[None, :, :]).to(device)
#     stu_kpts = stu_kpts.expand(stu_kpts.shape[0], tch_kpts.shape[1],
#                                stu_kpts.shape[2], 3)
#     tch_kpts = tch_kpts.expand(stu_kpts.shape[0], tch_kpts.shape[1],
#                                stu_kpts.shape[2], 3)

#     # (N, M, 17, 3, 2)
#     matrix = torch.stack((stu_kpts, tch_kpts), dim=4)

#     # TODO : consider some other sort of masking ????
#     # mask = torch.logical_and(matrix[:, :, :, 2, 0] > 0.3,
#     #                          matrix[:, :, :, 2, 1] > 0.3)
#     # matrix[~mask] = 0.0
#     matrix_ = matrix.clone()
#     print(matrix_.shape)
#     # matrix_[matrix == 0] = 256
#     x_min = matrix_.narrow(3, 0, 1).min(dim=2).values
#     y_min = matrix_.narrow(3, 1, 1).min(dim=2).values
#     z_min = matrix_.narrow(3, 2, 1).min(dim=2).values
#     matrix_ = matrix.clone()
#     # matrix_[matrix == 0] = 0
#     x_max = matrix_.narrow(3, 0, 1).max(dim=2).values
#     y_max = matrix_.narrow(3, 1, 1).max(dim=2).values
#     z_max = matrix_.narrow(3, 2, 1).max(dim=2).values

#     matrix_ = matrix.clone()
#     matrix_[:, :, :, 0] = (matrix_[:, :, :, 0] - x_min) / (
#         x_max - x_min + 1e-4)
#     matrix_[:, :, :, 1] = (matrix_[:, :, :, 1] - y_min) / (
#         y_max - y_min + 1e-4)
#     matrix_[:, :, :, 2] = (matrix_[:, :, :, 2] - z_min) / (
#         z_max - z_min + 1e-4)
    
    
#     xyz_dist = matrix_[..., :, 0] - matrix_[..., :, 1] # (N, M, 17, 3)
#     # score = matrix_[..., 2, 0] * matrix_[..., 2, 1]

#     # similarity = (torch.exp(-50 * xyz_dist.pow(2).sum(dim=-1)) *
#     #               score).sum(dim=-1) / (
#     #                   score.sum(dim=-1) + 1e-6)
#     # num_visible_kpts = score.sum(dim=-1)
#     # similarity = similarity * torch.log(
#     #     (1 + (num_visible_kpts - 1) * 10).clamp(min=1)) / np.log(161)

#     similarity = torch.nn.functional.softmax(
#         -50 * xyz_dist.pow(2).sum(dim=(-1, -2)), 
#         dim=-1
#     )

#     similarity[similarity.isnan()] = 0

#     plt.figure(figsize=(10, 8))
#     plt.imshow(xyz_dist.pow(2).sum(dim=(-1, -2)).cpu().numpy(), cmap='viridis', aspect='equal', origin='upper')
#     plt.colorbar(label='Value')

#     plt.xticks(np.arange(0, 301, 50))
#     plt.yticks(np.arange(0, 301, 50))

#     plt.xlim(-0.5, 300.5)
#     plt.ylim(300.5, -0.5)  # Reverse y-axis to start from 0 at the top

#     # Add title and labels
#     plt.xlabel('Column Index')
#     plt.ylabel('Row Index')
#     plt.savefig("cost.png")

#     return similarity



# @torch.no_grad()
# def calculate_similarity(tch_kpts: np.ndarray, stu_kpts: np.ndarray):
#     assert tch_kpts.shape[1] == 17
#     assert tch_kpts.shape[2] == 3
#     assert stu_kpts.shape[1] == 17
#     assert stu_kpts.shape[2] == 3

#     similarity1 = _calculate_similarity(tch_kpts, stu_kpts)

#     # flip_indices = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

#     # stu_kpts_flip = stu_kpts[:, flip_indices]
#     # stu_kpts_flip[..., 0] = 191.5 - stu_kpts_flip[..., 0]
#     # similarity2 = _calculate_similarity(tch_kpts, stu_kpts_flip)

#     # similarity = torch.stack((similarity1, similarity2)).max(dim=0).values

#     return similarity1


# @torch.no_grad()
# def select_piece_from_similarity(similarity):
#     m, n = similarity.size()
#     row_indices = torch.arange(m).view(-1, 1).expand(m, n).to(similarity)
#     col_indices = torch.arange(n).view(1, -1).expand(m, n).to(similarity)
#     diagonal_indices = similarity.size(0) - 1 - row_indices + col_indices
#     unique_diagonal_indices, inverse_indices = torch.unique(
#         diagonal_indices, return_inverse=True)

#     diagonal_sums_list = torch.zeros(
#         unique_diagonal_indices.size(0),
#         dtype=similarity.dtype,
#         device=similarity.device)
#     diagonal_sums_list.scatter_add_(0, inverse_indices.view(-1),
#                                     similarity.view(-1))
#     diagonal_sums_list[:min(m, n) // 4] = 0
#     diagonal_sums_list[-min(m, n) // 4:] = 0
#     index = diagonal_sums_list.argmax().item()

#     similarity_smooth = torch.nn.functional.max_pool2d(
#         similarity[None], (1, 11), stride=(1, 1), padding=(0, 5))[0]
#     similarity_vec = similarity_smooth.diagonal(offset=index - m +
#                                                 1).cpu().numpy()

#     stu_start = max(0, m - 1 - index)
#     tch_start = max(0, index - m + 1)

#     return dict(
#         stu_start=stu_start,
#         tch_start=tch_start,
#         length=len(similarity_vec),
#         similarity=similarity_vec)





class Skeleton: 
    
    def __init__(self, motion, view): 

        assert motion.shape[1] == 17 and motion.shape[2] == 3
        

        self.parent = view.scene 
        lines = scene.Line(parent=view.scene, color='gray', method='gl', width=20)
        scatter = scene.Markers(parent=view.scene, edge_color='green')
        
        colors = np.full((len(JOINT_PAIRS), 3), COLOR_MID)
        colors[np.isin(JOINT_PAIRS, JOINT_PAIRS_LEFT).all(axis=1)] = COLOR_LEFT
        colors[np.isin(JOINT_PAIRS, JOINT_PAIRS_RIGHT).all(axis=1)] = COLOR_RIGHT
        lines.set_data(color=np.repeat(colors, 2, axis=0))

        self.frame_text = scene.Text(f'', color='white', font_size=80, parent=view.scene)
        
        self.lines = lines
        self.scatter = scatter

        self.frame_idx = 0 
        # translate to root 
        preprocessed_motion = self._preprocess_motion(motion)
        self.motion = self.translate_to_origin(preprocessed_motion, joint_idx=0) # (N, 17, 3)
        self.max_frame = len(self.motion)


    def _preprocess_motion(self, motion): 
        _motion = copy.deepcopy(motion)
        _motion[:, :, 0] = - motion[:, :, 0]
        _motion[:, :, 1] = - motion[:, :, 2]
        _motion[:, :, 2] = - motion[:, :, 1]
        return _motion

    def translate_to_origin(self, motion, joint_idx): 
        translated_motion = motion - np.reshape(motion[0, joint_idx, :], (1, 1, -1))
        return translated_motion
    
    def update(self):
        if self.frame_idx < self.max_frame:
            j3d = self.motion[self.frame_idx, :, :] # (17, 3)
            # Update joint positions
            self.scatter.set_data(j3d, edge_color='black', face_color='white', size=10)
            
            # Update limb positions
            connects = np.c_[j3d[JOINT_PAIRS[:, 0]], j3d[JOINT_PAIRS[:, 1]]].reshape(-1, 3)
            self.lines.set_data(pos=connects, connect='segments')

            # update text 
            highest_point = np.max(j3d[:, 2])
            self.frame_text.text = f'Frame: {self.frame_idx}'
            self.frame_text.pos = [j3d[0, 0], j3d[0, 1], highest_point + 0.1]

            # update frame index 
            self.frame_idx += 1

    def __len__(self): 
        return len(self.motion)


        




class ComparisonWorld: 

    def __init__(self, teacher_motion, student_motion, fps=30, y_vertical=True): 
        self.fps = fps 
        self.y_vertical = y_vertical

        self._setup_world(teacher_motion, student_motion)


    def _setup_world(self, teacher_motion, student_motion): 
        canvas = scene.SceneCanvas(show=True)
        view = canvas.central_widget.add_view()
        camera = scene.cameras.TurntableCamera(elevation=0, azimuth=0)
        view.camera = camera
        self.view = view 
        self.camera = camera

        tch_skeleton = Skeleton(teacher_motion, view)
        self.tch_skeleton = tch_skeleton
        std_skeleton = Skeleton(student_motion, view)
        self.std_skeleton = std_skeleton

        self.curr_frame = 0 
        self.max_frame = max(len(tch_skeleton), len(std_skeleton))

        self.timer = app.Timer(interval=1/self.fps, connect=self._on_timer, start=True)

    def _on_timer(self, event):
        if self.curr_frame >= self.max_frame:
            self.curr_frame = 0

        self.tch_skeleton.update()
        # self.std_skeleton.update()

        self.curr_frame += 1


    def _display_axis(self): 
        axis_length = 4.0
        axis_points = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])

        axis_connects = np.array([[0, 1], [0, 2], [0, 3]])

        axis_colors = np.array([
            [1, 1, 1, 1],  # Red,
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1]   # Blue
        ])

        axis_lines = scene.Line(pos=axis_points, connect=axis_connects, color=axis_colors,
                            method='gl', parent=self.view.scene, width=5)
        
        # Add axis labels
        axis_labels = ['X', 'Y', 'Z']
    
        label_offset = 0.1  # Offset to position labels slightly away from axis ends
        for i, label in enumerate(axis_labels):
            pos = axis_points[i+1] + label_offset
            text = scene.Text(label, pos=pos, color=axis_colors[i+1],
                            font_size=60, parent=self.view.scene)


    def translate_to_origin(self, motion, joint_idx): 
        assert motion.shape[1] == 17 and motion.shape[2] == 3
        translated_motion = motion - np.reshape(motion[0, joint_idx, :], (1, 1, -1))
        return translated_motion

    def display(self): 
        self._display_axis()
        app.run()



if __name__ == "__main__": 

    opts = parse_args() 

    student_motion = load_skeleton(opts.student_motion)[30:]
    teacher_motion = load_skeleton(opts.teacher_motion)

    # compute similarity  
    distance, path = dtw_skeleton(teacher_motion, student_motion)
    print(path)
    # translate to origin 

    world = ComparisonWorld(student_motion, teacher_motion) 

    world.display()

    




