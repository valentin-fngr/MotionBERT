from tqdm import tqdm 
from torch.utils.data import DataLoader
from lib.data.dataset_wild import WildDetDataset
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data

class Skeleton3DInference: 

    def __init__(self, config, checkpoint, device="cpu"): 
        self.config = config 
        args = get_config(config)
        self.args = args
        self.device = device

        self.model_pose = self._load_model(args, checkpoint)
        
        self.testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
        }   

    def _load_model(self, args, checkpoint): 
        model_backbone = load_backbone(self.args)
        backbone_checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict({k[7:]: v for k,v in backbone_checkpoint['model_pos'].items()}, strict=True)
        model_backbone.to(self.device)
        model_backbone.eval()
        return model_backbone

    @torch.no_grad()
    def inference(self, skeleton_2d, vid_size, focus=1):
        wild_dataset = WildDetDataset(skeleton_2d, clip_len=180, vid_size=vid_size, scale_range=[1,1], focus=focus)
        test_loader = DataLoader(wild_dataset, **self.testloader_params)
        results_all = []
        print(f"INFO : running infernece on device = {self.device}")
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            batch_input = batch_input

            if self.args.no_conf:
                batch_input = batch_input[:, :, :, :2]

            if self.args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = self.model_pose(batch_input) # [B, F, J, dim_feat] 
                predicted_3d_pos_flip = self.model_pose(batch_input_flip) 
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else: 
                predicted_3d_pos = self.model_pose(batch_input)
        
            if self.args.rootrel: 
                predicted_3d_pos = predicted_3d_pos - predicted_3d_pos[:, :, 0:1, :]
                predicted_3d_pos[:,:,0,:]=0         
            else: 
                predicted_3d_pos[:,0,0,2]=0

            results_all.append(predicted_3d_pos)

        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)
        return results_all 
