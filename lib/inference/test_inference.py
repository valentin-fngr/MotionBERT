import json 
import sys 
import os 
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)
print(sys.path)

from inference_api import Skeleton3DInference 
from lib.utils.vismo import render_and_save


DEBUG = True
CONFIG = "configs/pose3d/MB_ft_h36m_global_lite.yaml" 
CHECKPOINT = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
DEVICE = "cpu"
SKELETON_2D_PATH = "inputs/AlphaPose_dance_3_people.json"
VID_SIZE = [360, 640]

with open(SKELETON_2D_PATH, "rb") as f: 
    skeleton_2d = json.load(f) 


inference_3d = Skeleton3DInference(CONFIG, CHECKPOINT, DEVICE)
skeleton_3d = inference_3d.inference(skeleton_2d, vid_size=VID_SIZE, focus=3)

if DEBUG: 
    render_and_save(skeleton_3d, 'outputs/X3D.mp4', keep_imgs=False, fps=30)
