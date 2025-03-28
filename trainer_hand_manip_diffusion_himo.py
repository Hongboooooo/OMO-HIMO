import torch
import argparse
import os
import yaml
import wandb
import numpy as np

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

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=300,
        results_folder='./results',
        load_folder='',
        use_wandb=True,  
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(device='cuda', enabled=amp)

        self.results_folder = results_folder

        self.load_folder = load_folder

        self.opt = opt 

        self.window = 120 # default 120

        self.num_objs = 2

        self.prep_dataloader()

        self.add_hand_processing = False

        self.retrain = opt.retrain

        self.load_milestone = opt.milestone

        
    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.load_folder, 'model-'+str(self.load_milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])
        self.optimizer.load_state_dict(data['optimizer'])

    def prep_dataloader(self):
        # Define dataset
        train_dataset = OMODataset(data_type="train", num_objs=self.num_objs)
        val_dataset = OMODataset(data_type="validation", num_objs=self.num_objs)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2))


    def train(self):

        if self.retrain == True:
            self.load(pretrained_path=None)

        init_step = self.step # initial self.step is 0

        print(f"Start from step {init_step}")

        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)

                data = data_dict['hands_positions_seq'].cuda() # BS X T X (22*3+22*6)
                batch_size, num_frames, num_hands, xyz = data.shape 
                data = data.reshape(batch_size, num_frames, num_hands*xyz)
                    
                objA_bps_data = data_dict['bps_object_geo_seq_o1'].cuda()
                objA_bps_data = objA_bps_data.reshape(objA_bps_data.size(0), objA_bps_data.size(1),-1)
                objA_center_pos = data_dict['obj_center_seq_o1'].cuda() # BS X T X 3 
                objA_data_cond = torch.cat((objA_center_pos, objA_bps_data), dim=-1) # BS X T X (3+1024*3)
                objA_data_cond = objA_data_cond.to(torch.float32)

                objB_bps_data = data_dict['bps_object_geo_seq_o2'].cuda()
                objB_bps_data = objB_bps_data.reshape(objB_bps_data.size(0), objB_bps_data.size(1),-1)
                objB_center_pos = data_dict['obj_center_seq_o2'].cuda() # BS X T X 3 
                objB_data_cond = torch.cat((objB_center_pos, objB_bps_data), dim=-1) # BS X T X (3+1024*3)
                objB_data_cond = objB_data_cond.to(torch.float32)

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                # explain:
                # assume:
                #   self.window = 3
                #   data.shape[0] = 2
                #   actual_seq_len = [2,3]
                # then:
                #   torch.arange(self.window+1) = [0,1,2,3]
                #   torch.arange(self.window+1).expand(data.shape[0], self.window+1) = [[0,1,2,3],[0,1,2,3]]
                #   actual_seq_len[:, None] = [[2],[3]]
                #   actual_seq_len[:, None].repeat(1, self.window+1) = [[2,2,2,2],[3,3,3,3]]
                #   tmp_mask = [[T,T,F,F],[T,T,T,F]]

                with autocast(device_type='cuda', enabled = self.amp): 
                # enable mix precision training, let pytorch decide to use FP16 or FP32
                # FP16 for saving memory and faster computation
                # FP32 for preciser computation and numerical stability

                    loss_diffusion = self.model(data, objA_data_cond, objB_data_cond, cond_mask=None, padding_mask=padding_mask)
                    
                    loss = loss_diffusion

                    if torch.isnan(loss).item():# mix precision is numerical instable. If computed loss has NaN, then skip this data
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                    # mixed precision can lead to underflow(very small number becomes Zero when using FP16)
                    # .scale(loss / 2) up scales the loss value to prevent underflow
                    # .backward() computes the gradients of the loss w.r.t parameters of model

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None] # collect all the parameters that have computed the gradient
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    # compute total gradient norm
                    # .detach() create a new tensor that share the same value as the original data but doesn't track its gradient to save memory
                    # First torch.norm(:, 2.0): compute L2 norm of the gradients of each parameter
                    # Second torch.norm(:, 2.0): compute L2 norm of stacked gradients' norm
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb: # log
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(), # loss is a tenosr, loss.item() is a float
                        }
                        wandb.log(log_dict)

                    if idx % 10 == 0 and i == 0: # log
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists: # NaN exists in model's gradients, then don't update in this time
                continue

            self.scaler.step(self.optimizer)
            # automatically implicitly unscaling the scaled gradients
            # update model's parameters using unscaled gradients
            self.scaler.update() 
            # update the scaler for next iteration basing on observed gradients. 
            # If the computed gradients are too large or too small, then scaling factors are adjusted to prevent overflow or underflow

            self.ema.update()
            # ema weights are updated like: ema_w = beta * ema_w + (1 - beta) * current_w
            # ema weights are used for inference

            if self.step != init_step and self.step % 10 == 0: # evaluate in every 10 steps
                self.ema.ema_model.eval()
                # set EMA_model to evaluation mode
                # EMA_model is the diffusion model 

                with torch.no_grad():

                # evaluation
                    val_data_dict = next(self.val_dl)

                    val_data = val_data_dict['hands_positions_seq'].cuda()
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

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_data, objA_data_cond, objB_data_cond, cond_mask=None, padding_mask=padding_mask)
                    # actual model is used for validation
                    val_loss = val_loss_diffusion 
                    if self.use_wandb: # log
                        val_log_dict = {
                            "Validation/Loss/Validation Loss": val_loss.item(),
                        }
                        wandb.log(val_log_dict)

                    # Save weights
                    milestone = self.step // self.save_and_sample_every
                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

            self.step += 1 # used for evaluation

        print('training complete')

        if self.use_wandb: # log
            wandb.run.finish()



def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=2e-5, help="this is learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="this is batch size") # original 32

    parser.add_argument('--dim_model', type=int, default=512) # original 512
    parser.add_argument('--num_dec_layers', type=int, default=4)
    parser.add_argument("--num_head", type=int, default=4, help="number of attention heads") # original 4
    parser.add_argument('--dim_key', type=int, default=256) # original 256
    parser.add_argument('--dim_value', type=int, default=256) # original 256

    parser.add_argument('--project', default='omo/train', help='output folder for weights and visualizations')
    parser.add_argument('--exp_name', default='stage1_exp_out_modified', help='save to project/exp_name')
    parser.add_argument('--wandb_pj_name', type=str, default='omo_baseline', help='wandb project name')
    parser.add_argument('--entity', default='HongboTeam', help='W&B entity')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--retrain', action="store_true", default=False, help='add to activate retrain mode')

    opt = parser.parse_args()

    return opt

def run_train(args, device):

    # Prepare Directories
    save_dir = Path(args.save_dir)
    weight_dir = save_dir / 'weights'
    weight_dir.mkdir(parents=True, exist_ok=True)
    load_dir = Path(args.load_dir)
    load_dir = load_dir / 'weights'

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)

    represent_dim = 2*3

    diffusion_model = CondGaussianDiffusion(args, d_feats=represent_dim, d_model=args.dim_model, \
                n_dec_layers=args.num_dec_layers, n_head=args.num_head, d_k=args.dim_key, d_v=args.dim_value, \
                max_timesteps=120+1, out_dim=represent_dim, timesteps=1000, \
                objective="pred_x0", loss_type="l1", \
                batch_size=args.batch_size)
    
    diffusion_model.to(device)

    trainer = Trainer(
        args,
        diffusion_model,
        train_batch_size=args.batch_size, # 32
        train_lr=args.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(weight_dir),
        load_folder=str(load_dir),
    )

    trainer.train()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_args = parse_arg()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    if main_args.retrain == True:
        main_args.milestone = 1
        main_args.weight_name = "stg1_2025-03-18 16:59:35"
        main_args.exp_name = f"re_{main_args.weight_name}_{main_args.milestone}_{now}"
        main_args.save_dir = os.path.join(main_args.project, main_args.exp_name)
        main_args.load_dir = os.path.join(main_args.project, main_args.weight_name)
        print(">>>Activate Retrain Mode<<<")
        print("Load weights from:",main_args.load_dir)
        print("Save Dir:",main_args.save_dir)
    else:
        main_args.milestone = 0
        main_args.exp_name = f"stg1_{now}"
        main_args.save_dir = os.path.join(main_args.project, main_args.exp_name)
        main_args.load_dir = ""
        
        print(">>>Start Train Mode<<<")
        print("Save Dir:",main_args.save_dir)

    run_train(args=main_args, device=main_device)

