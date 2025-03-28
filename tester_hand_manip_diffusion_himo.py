import torch
import argparse
import os
import yaml
import numpy as np
import shutil

from manip.model.transformer_hand_manip_cond_diffusion_model_himo import CondGaussianDiffusion
from pathlib import Path
from torch.optim import Adam
from ema_pytorch import EMA
from torch.amp import autocast, GradScaler
from torch.utils import data
from manip.data.hand_dataset_omo import OMODataset
from datetime import datetime

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Tester(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        test_batch_size=8,
        train_lr=1e-4,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=10000,
        weights_folder='./results',
    ):
        super().__init__()

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = test_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(device='cuda', enabled=amp)

        self.weights_folder = weights_folder

        self.opt = opt 

        self.window = 120 # default 120

        self.num_objs = 2

        self.prep_dataloader()

        self.top_n = 10

        self.test_on_train = False

        self.add_hand_processing = False

        self.for_quant_eval = True

        self.objs_points = np.load("himo_data/object_bps.npz", allow_pickle=True)

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.weights_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])
    
    def prep_dataloader(self):
        # Define dataset
        train_dataset = OMODataset(data_type="train", num_objs=self.num_objs)
        val_dataset = OMODataset(data_type="validation", num_objs=self.num_objs)

        self.ds = train_dataset 
        self.val_ds = val_dataset

    def compute_s1_metrics(self, ori_jpos_pred, ori_jpos_gt):
        # pred_hand_jpos: T X 2 X 3
        # gt_hand_jpos: T X 2 X 3 

        ori_jpos_pred = ori_jpos_pred.reshape(-1, 2, 3)
        ori_jpos_gt = ori_jpos_gt.reshape(-1, 2, 3)

        lhand_idx = 0
        rhand_idx = 1
        lhand_jpos_pred = ori_jpos_pred[:, lhand_idx, :].detach().cpu().numpy() 
        rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy() 
        lhand_jpos_gt = ori_jpos_gt[:, lhand_idx, :].detach().cpu().numpy()
        rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy() 
        lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=1).mean() * 1000
        rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=1).mean() * 1000
        hand_jpe = (lhand_jpe+rhand_jpe)/2.0 

        return lhand_jpe, rhand_jpe, hand_jpe 
    
    def compute_metrics(self, hands_jpos_gt, hands_jpos_pred, obj_verts, actual_len, use_joints24=True): # need modify
        # verts_gt: T X Nv X 3 
        # jpos_gt: T X J X 3
        # gt_trans: T X 3
        # gt_rot_mat: T X 22 X 3 X 3 
        # human_faces: Nf X 3, array  
        # obj_verts: T X No X 3
        # obj_faces: Nf X 3, array  
        # actual_len: scale value 

        hands_jpos_gt = hands_jpos_gt[:actual_len]
        hands_jpos_pred = hands_jpos_pred[:actual_len]
        obj_verts = obj_verts[:actual_len]

        # Compute contact score 
        num_obj_verts = obj_verts.shape[1]

        if use_joints24:
            contact_threh = 0.05
        else:
            contact_threh = 0.10 

        gt_lhand_jnt = hands_jpos_gt[:,:3] # T X 3 
        gt_rhand_jnt = hands_jpos_gt[:,3:] # T X 3 

        # print("gt_lhand_jnt:", gt_lhand_jnt.shape)

        # print("gt_lhand_jnt.repeat(1, num_obj_verts, 1):", gt_lhand_jnt[:,None,:].repeat(1, num_obj_verts, 1).shape)

        gt_lhand2obj_dist = torch.sqrt(((gt_lhand_jnt[:,None,:].repeat(1, num_obj_verts, 1) - obj_verts.to(gt_lhand_jnt.device))**2).sum(dim=-1)) # T X N  
        gt_rhand2obj_dist = torch.sqrt(((gt_rhand_jnt[:,None,:].repeat(1, num_obj_verts, 1) - obj_verts.to(gt_rhand_jnt.device))**2).sum(dim=-1)) # T X N  

        gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0] # T
        gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0] # T

        gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threh)
        gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)

        lhand_jnt = hands_jpos_pred[:,:3] # T X 3 
        rhand_jnt = hands_jpos_pred[:,3:] # T X 3 

        lhand2obj_dist = torch.sqrt(((lhand_jnt[:,None,:].repeat(1, num_obj_verts, 1) - obj_verts.to(lhand_jnt.device))**2).sum(dim=-1)) # T X N  
        rhand2obj_dist = torch.sqrt(((rhand_jnt[:,None,:].repeat(1, num_obj_verts, 1) - obj_verts.to(rhand_jnt.device))**2).sum(dim=-1)) # T X N  
    
        lhand2obj_dist_min = lhand2obj_dist.min(dim=1)[0] # T
        rhand2obj_dist_min = rhand2obj_dist.min(dim=1)[0] # T

        lhand_contact = (lhand2obj_dist_min < contact_threh)
        rhand_contact = (rhand2obj_dist_min < contact_threh)

        num_steps = hands_jpos_gt.shape[0]

        # Compute the distance between hand joint and object for frames that are in contact with object in GT. 
        contact_dist = 0
        gt_contact_dist = 0 

        gt_contact_cnt = 0
        for idx in range(num_steps):
            if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
                gt_contact_cnt += 1 

                contact_dist += min(lhand2obj_dist_min[idx], rhand2obj_dist_min[idx])
                gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx])

        if gt_contact_cnt == 0:
            contact_dist = 0 
            gt_contact_dist = 0 
        else:
            contact_dist = contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)
            gt_contact_dist = gt_contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)

        # Compute precision and recall for contact. 
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for idx in range(num_steps):
            gt_in_contact = (gt_lhand_contact[idx] or gt_rhand_contact[idx]) 
            pred_in_contact = (lhand_contact[idx] or rhand_contact[idx])
            if gt_in_contact and pred_in_contact:
                TP += 1

            if (not gt_in_contact) and pred_in_contact:
                FP += 1

            if (not gt_in_contact) and (not pred_in_contact):
                TN += 1

            if gt_in_contact and (not pred_in_contact):
                FN += 1

        contact_acc = (TP+TN)/(TP+FP+TN+FN)

        if (TP+FP) == 0: # Prediction no contact!!!
            contact_precision = 0
            print("Contact precision, TP + FP == 0!!")
        else:
            contact_precision = TP/(TP+FP)
        
        if (TP+FN) == 0: # GT no contact! 
            contact_recall = 0
            print("Contact recall, TP + FN == 0!!")
        else:
            contact_recall = TP/(TP+FN)

        if contact_precision == 0 and contact_recall == 0:
            contact_f1_score = 0 
        else:
            contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall) 
    
        return gt_contact_dist, contact_dist, contact_precision, contact_recall, contact_acc, contact_f1_score  

    def test(self):
        self.load(milestone=self.opt.milestone, pretrained_path=self.opt.checkpoint)
        self.ema.ema_model.eval()

        s1_global_hand_jpe_list = [] 
        s1_global_lhand_jpe_list = []
        s1_global_rhand_jpe_list = [] 

        contact_precision_list = []
        contact_recall_list = [] 

        contact_acc_list = []
        contact_f1_score_list = []

        gt_contact_dist_list = []
        contact_dist_list = []

        val_seq_names_list = []

        test_dir = Path(self.opt.test_dir)
        result_dir = test_dir / "InferredResults"
        result_dir.mkdir(parents=True, exist_ok=True)
        best_result_dir = test_dir / "BestResults"
        best_result_dir.mkdir(parents=True, exist_ok=True)
        worst_result_dir = test_dir / "WorstResults"
        worst_result_dir.mkdir(parents=True, exist_ok=True)

        print("result_dir", result_dir)
        print("best_result_dir", best_result_dir)
        print("worst_result_dir", worst_result_dir)

        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 

        if self.for_quant_eval:
            num_samples_per_seq = 20 # default 20
        else:
            num_samples_per_seq = 1

        print("test_on_train:", self.test_on_train)
        print("for_quant_eval:", self.for_quant_eval)
        print("num_samples_per_seq:", num_samples_per_seq)

        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):

                if (not s_idx % 8 == 0) and (not self.for_quant_eval): # Visualize part of data
                    continue 

                # if (not s_idx % 8 == 0): # for test
                #     continue 

                val_data = val_data_dict['hands_positions_seq'].cuda() # BS X T X (22*3+22*6)
                batch_size, num_frames, num_hands, xyz = val_data.shape
                val_data = val_data.reshape(batch_size, num_frames, num_hands*xyz)
                    
                objA_bps_data = val_data_dict['bps_object_geo_seq_o1'].cuda()
                objA_bps_data = objA_bps_data.reshape(objA_bps_data.size(0), objA_bps_data.size(1),-1)
                objA_center_pos = val_data_dict['obj_center_seq_o1'].cuda() # BS X T X 3
                objA_data_cond = torch.cat((objA_center_pos, objA_bps_data), dim=-1) # BS X T X (3+1024*3)
                objA_data_cond = objA_data_cond.to(torch.float32)

                objB_bps_data = val_data_dict['bps_object_geo_seq_o2'].cuda()
                objB_bps_data = objB_bps_data.reshape(objB_bps_data.size(0), objB_bps_data.size(1),-1)
                objB_center_pos = val_data_dict['obj_center_seq_o2'].cuda() # BS X T X 3
                objB_data_cond = torch.cat((objB_center_pos, objB_bps_data), dim=-1) # BS X T X (3+1024*3)
                objB_data_cond = objB_data_cond.to(torch.float32)

                o1_rotations = val_data_dict['rotation_seq_o1'].cpu().numpy()
                o2_rotations = val_data_dict['rotation_seq_o2'].cpu().numpy()
                o1_translations = val_data_dict['translation_seq_o1'].cpu().numpy()
                o2_translations = val_data_dict['translation_seq_o2'].cpu().numpy()

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                # Objects_names
                o1_names = val_data_dict["o1_name"]
                o2_names = val_data_dict["o2_name"]

                s1_lhand_jpe_per_seq = []
                s1_rhand_jpe_per_seq = []
                s1_hand_jpe_per_seq = []
                contact_precision_per_seq = []
                contact_recall_per_seq = []
                contact_acc_per_seq = []
                contact_f1_score_per_seq = []
                gt_contact_dist_per_seq = []
                contact_dist_per_seq = []
                val_seq_names = val_data_dict['seq_name']

                actual_len_list = val_data_dict['seq_len']

                for sample_idx in range(num_samples_per_seq):
                    # Stage 1 
                    pred_hand_foot_jpos = self.ema.ema_model.sample(val_data, objA_data_cond, objB_data_cond, cond_mask=None, padding_mask=padding_mask) # progress bar is in here
                    # print("pred_hand_foot_jpos:", pred_hand_foot_jpos.shape)
                    # print("pred_hand_foot_jpos:", val_data_dict['seq_name'])
                    inference_out = pred_hand_foot_jpos.cpu().numpy()
                    for bi in range(batch_size):
                        np.save(f"{result_dir}/{val_data_dict['seq_name'][bi]}.npy", inference_out[bi])

                    for s1_s_idx in range(batch_size): 
                        pred_eval = pred_hand_foot_jpos[s1_s_idx, :actual_len_list[s1_s_idx]]
                        val_eval = val_data[s1_s_idx, :actual_len_list[s1_s_idx]]


                        s1_lhand_jpe, s1_rhand_jpe, s1_hand_jpe = self.compute_s1_metrics(pred_eval, val_eval)
                      
                        s1_lhand_jpe_per_seq.append(s1_lhand_jpe)
                        s1_rhand_jpe_per_seq.append(s1_rhand_jpe)
                        s1_hand_jpe_per_seq.append(s1_hand_jpe)


                        o1_vertices = self.objs_points[o1_names[s1_s_idx]]
                        o2_vertices = self.objs_points[o2_names[s1_s_idx]]

                        o1_vertices_seq = np.repeat(o1_vertices, actual_len_list[s1_s_idx], axis=0)
                        rotated_o1_vertices_seq = np.matmul(o1_vertices_seq, o1_rotations[s1_s_idx, :actual_len_list[s1_s_idx]].transpose(0,2,1))
                        translated_o1_vertices_seq = rotated_o1_vertices_seq + o1_translations[s1_s_idx, :actual_len_list[s1_s_idx]][:,None,:]

                        o2_vertices_seq = np.repeat(o2_vertices, actual_len_list[s1_s_idx], axis=0)
                        rotated_o2_vertices_seq = np.matmul(o2_vertices_seq, o2_rotations[s1_s_idx, :actual_len_list[s1_s_idx]].transpose(0,2,1))
                        translated_o2_vertices_seq = rotated_o2_vertices_seq + o2_translations[s1_s_idx, :actual_len_list[s1_s_idx]][:,None,:]

                        obj_verts_list = np.concatenate((translated_o1_vertices_seq, translated_o2_vertices_seq), axis=1)
                        concatenated_objs_verts = torch.from_numpy(obj_verts_list)

                        gt_contact_dist, contact_dist, contact_precision, contact_recall, contact_acc, contact_f1_score = \
                        self.compute_metrics(val_eval, pred_eval, concatenated_objs_verts, actual_len_list[s1_s_idx], use_joints24=True)
                        
                        contact_precision_per_seq.append(contact_precision)
                        contact_recall_per_seq.append(contact_recall)

                        contact_acc_per_seq.append(contact_acc) 
                        contact_f1_score_per_seq.append(contact_f1_score) 

                        gt_contact_dist_per_seq.append(gt_contact_dist)
                        contact_dist_per_seq.append(contact_dist)

                # n samples each batch is accomplished above

                if self.for_quant_eval:
                    s1_lhand_jpe_per_seq = np.asarray(s1_lhand_jpe_per_seq).reshape(num_samples_per_seq, batch_size)
                    s1_rhand_jpe_per_seq = np.asarray(s1_rhand_jpe_per_seq).reshape(num_samples_per_seq, batch_size)
                    s1_hand_jpe_per_seq = np.asarray(s1_hand_jpe_per_seq).reshape(num_samples_per_seq, batch_size)

                    contact_precision_per_seq = np.asarray(contact_precision_per_seq).reshape(num_samples_per_seq, batch_size) 
                    contact_recall_per_seq = np.asarray(contact_recall_per_seq).reshape(num_samples_per_seq, batch_size) 

                    contact_acc_per_seq = np.asarray(contact_acc_per_seq).reshape(num_samples_per_seq, batch_size) 
                    contact_f1_score_per_seq = np.asarray(contact_f1_score_per_seq).reshape(num_samples_per_seq, batch_size) 

                    gt_contact_dist_per_seq = np.asarray(gt_contact_dist_per_seq).reshape(num_samples_per_seq, batch_size)
                    contact_dist_per_seq = np.asarray(contact_dist_per_seq).reshape(num_samples_per_seq, batch_size) 

                    best_sample_idx = s1_hand_jpe_per_seq.argmin(axis=0) # sample_num 

                    s1_hand_jpe = s1_hand_jpe_per_seq[best_sample_idx, list(range(batch_size))]
                    s1_lhand_jpe = s1_lhand_jpe_per_seq[best_sample_idx, list(range(batch_size))]
                    s1_rhand_jpe = s1_rhand_jpe_per_seq[best_sample_idx, list(range(batch_size))]

                    contact_precision_seq = contact_precision_per_seq[best_sample_idx, list(range(batch_size))]
                    contact_recall_seq = contact_recall_per_seq[best_sample_idx, list(range(batch_size))] 
                    
                    contact_acc_seq = contact_acc_per_seq[best_sample_idx, list(range(batch_size))]
                    contact_f1_score_seq = contact_f1_score_per_seq[best_sample_idx, list(range(batch_size))]

                    gt_contact_dist_seq = gt_contact_dist_per_seq[best_sample_idx, list(range(batch_size))]
                    contact_dist_seq = contact_dist_per_seq[best_sample_idx, list(range(batch_size))] 

                        
                    for tmp_seq_idx in range(batch_size):
                        s1_global_lhand_jpe_list.append(s1_lhand_jpe[tmp_seq_idx])
                        s1_global_rhand_jpe_list.append(s1_rhand_jpe[tmp_seq_idx])
                        s1_global_hand_jpe_list.append(s1_hand_jpe[tmp_seq_idx])

                        contact_precision_list.append(contact_precision_seq[tmp_seq_idx])
                        contact_recall_list.append(contact_recall_seq[tmp_seq_idx])

                        contact_acc_list.append(contact_acc_seq[tmp_seq_idx])
                        contact_f1_score_list.append(contact_f1_score_seq[tmp_seq_idx])

                        gt_contact_dist_list.append(gt_contact_dist_seq[tmp_seq_idx])
                        contact_dist_list.append(contact_dist_seq[tmp_seq_idx])

                        val_seq_names_list.append(val_seq_names[tmp_seq_idx])


        

        sorted_indices = np.argsort(s1_global_hand_jpe_list)
        worst_indices = sorted_indices[-(self.top_n):][::-1]
        best_indices = sorted_indices[:(self.top_n)]

        for i in range(self.top_n):
            # print(f"{result_dir}/{val_seq_names_list[worst_indices[i]]}.npy", "||", worst_result_dir)
            shutil.copy(f"{result_dir}/{val_seq_names_list[worst_indices[i]]}.npy", worst_result_dir)
            shutil.copy(f"{result_dir}/{val_seq_names_list[best_indices[i]]}.npy", best_result_dir)

        print("worst_indices:", [val_seq_names_list[wi] for wi in worst_indices])
        print("best_indices:", [val_seq_names_list[bi] for bi in best_indices])

        if self.for_quant_eval:
            s1_mean_hand_jpe = np.asarray(s1_global_hand_jpe_list).mean()
            s1_mean_lhand_jpe = np.asarray(s1_global_lhand_jpe_list).mean()
            s1_mean_rhand_jpe = np.asarray(s1_global_rhand_jpe_list).mean() 

            mean_contact_precision = np.asarray(contact_precision_list).mean()
            mean_contact_recall = np.asarray(contact_recall_list).mean() 

            mean_contact_acc = np.asarray(contact_acc_list).mean()
            mean_contact_f1_score = np.asarray(contact_f1_score_list).mean() 

            mean_gt_contact_dist = np.asarray(gt_contact_dist_list).mean()
            mean_contact_dist = np.asarray(contact_dist_list).mean()

            print("*****************************************Quantitative Evaluation*****************************************")
            print("The number of sequences: {0}".format(len(s1_global_hand_jpe_list)))
            print("Stage 1 Left Hand JPE: {0}, Stage 1 Right Hand JPE: {1}, Stage 1 Two Hands JPE: {2}".format(s1_mean_lhand_jpe, s1_mean_rhand_jpe, s1_mean_hand_jpe))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score))
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))



def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=2e-5, help="this is learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="this is batch size")

    parser.add_argument('--dim_model', type=int, default=512) # original 512
    parser.add_argument('--num_dec_layers', type=int, default=4)
    parser.add_argument("--num_head", type=int, default=4, help="number of attention heads") # original 4
    parser.add_argument('--dim_key', type=int, default=256) # original 256
    parser.add_argument('--dim_value', type=int, default=256) # original 256

    parser.add_argument('--train', default='omo/train', help='output folder for weights and visualizations')
    parser.add_argument('--inference', default='omo/test', help='folder for inferred results')
    parser.add_argument('--exp_name', default='stage1_exp_out_modified', help='save to project/exp_name')
    parser.add_argument('--entity', default='HongboTeam', help='W&B entity')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')

    opt = parser.parse_args()

    return opt

def run_test(args, device):

    # Prepare Directories
    train_dir = Path(args.train_dir)
    weight_dir = train_dir / 'weights'

    represent_dim = 2*3

    diffusion_model = CondGaussianDiffusion(args, d_feats=represent_dim, d_model=args.dim_model, \
                n_dec_layers=args.num_dec_layers, n_head=args.num_head, d_k=args.dim_key, d_v=args.dim_value, \
                max_timesteps=120+1, out_dim=represent_dim, timesteps=1000, \
                objective="pred_x0", loss_type="l1", \
                batch_size=args.batch_size)
    
    diffusion_model.to(device)

    tester = Tester(
        args,
        diffusion_model,
        test_batch_size=args.batch_size, # 32
        train_lr=args.learning_rate, # 1e-4
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        weights_folder=str(weight_dir),
    )

    tester.test()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_args = parse_arg()
    main_args.exp_name = "stg1_2025-03-16 21:32:27"
    main_args.milestone = 9
    main_args.train_dir = os.path.join(main_args.train, main_args.exp_name)
    main_args.test_dir = os.path.join(main_args.inference, main_args.exp_name)
    main_args.checkpoint = None

    print("Going to load model from:", main_args.train_dir)

    run_test(args=main_args, device=main_device)

