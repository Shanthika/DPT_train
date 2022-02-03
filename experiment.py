#contains the model call and training_one_epoch

import numpy as np
import torch
import math
import torch.optim as optim
from collections import Counter



import pytorch_lightning as pl
from dpt.models import DPTDepthModel
from dpt.loss import MaskedL1Loss, DepthMetrics



class DPT(pl.LightningModule):
    def __init__(self,config):

        super().__init__()
        self.load_ckpt = config['model']['load_ckpt']
        self.model_path = config['model']['model_path']
        self.lr = float(config['experiment']['learning_rate'])
        self.scale = config['experiment']['scale']
        self.shift = config['experiment']['shift']
        self.invert = config['experiment']['invert']
        self.backbone = config['experiment']['backbone']
        self.non_negative = config['experiment']['non_negative']
        self.enable_attention_hooks = config['experiment']['enable_attention_hooks']
        self.epochs = config['experiment']['epochs']
        self.verbose = config['experiment']['verbose']
        self.save_hyperparameters()
        self.metrics = DepthMetrics()
        self.s = []
        self.t = []
        self.gpus = config['experiment']['gpus'] 
        self.val_outputs = None
        self.criteria = MaskedL1Loss()

        self.model = DPTDepthModel(
            path=self.model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            load_ckpt=self.load_ckpt
        )
        # if(config['experiment']['gpus']>1):
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config['experiment']['gpu_ids']])
        


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        yhat = self.model(x)
        yhat = torch.unsqueeze(yhat,axis=1)

        loss = self.criteria(yhat, y)
        self.log('train_loss', loss, 
                 on_step=False,
                 on_epoch=True, 
                 rank_zero_only=True,
                 sync_dist=self.gpus> 1)
        
        # gather scale, shift from computed metrics
        # to use for validation later
        metrics = self.metrics(yhat.detach(), y.detach())
        
        self.s.append(metrics.pop('s'))
        self.t.append(metrics.pop('t'))
        
        for metric,val in metrics.items():
            self.log(metric, val,
                     on_step=False, 
                     on_epoch=True,
                     rank_zero_only=True,
                     sync_dist=self.gpus > 1)

        # wait until next release of lightning to update
        # self.log_dict(metrics,
        #               on_step=False,
        #               on_epoch=True,
        #               rank_zero_only=True,
        #               sync_dist=self.gpus> 1)
        
        return {'loss': loss, **metrics}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        yhat = self.model(x)
        yhat = torch.unsqueeze(yhat,axis=1)
        loss = self.criteria(yhat, y)
        self.log('val_loss', loss, 
                 on_step=False,
                 on_epoch=True,
                 rank_zero_only=True,
                 sync_dist= self.gpus> 1)
        
        metrics = self.metrics(yhat, y, (self.s, self.t) if type(self.s) is torch.Tensor else None)
        
        for metric,val in metrics.items():
            self.log(metric, val,
                     on_step=False, 
                     on_epoch=True,
                     rank_zero_only=True,
                     sync_dist=self.gpus> 1)
            
        return {'loss': loss, **metrics}
    
    def configure_optimizers(self):
        return optim.Adam([
                            {'params': filter(lambda p: p.requires_grad, self.model.pretrained.parameters())},
                            {'params': self.model.scratch.parameters(), 'lr': self.lr * 10}
                          ], 
                          lr=self.lr)
    
    def training_epoch_end(self, epoch_outputs):
        res = Counter()
        for out in epoch_outputs:
            res += out
        self.print(f'--- Epoch {self.current_epoch} training ---')
        for name, val in res.items():
            if name == 'loss':
                name = 'train_loss'
                val = val.item()
            if math.isinf(val):
                self.print(f'{name}: {val}')
                continue
        self.print('-'*25)
        self.logger.log_graph(self)
        
    def validation_epoch_end(self, epoch_outputs):
        self.val_outputs = epoch_outputs
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
    def on_train_epoch_start(self):
        self.s, self.t = [], []

        if self.verbose:
            self.print(f'Epoch {self.current_epoch}')
            
    def on_train_epoch_end(self):
        res = Counter()
        for out in self.val_outputs:
            res += out
        self.print(f'--- Epoch {self.current_epoch} validation ---')
        for name, val in res.items():
            if name == 'loss': 
                name = 'val_loss'
                val = val.item()
            if math.isinf(val):
                self.print(f'{name}: {val}')
                continue
        self.print('-'*25)
        self.logger.log_graph(self)

        self.val_outputs = None
            
    def on_validation_epoch_start(self):
        if len(self.s) > 0:
            self.s, self.t = torch.cat(self.s).mean(0), torch.cat(self.t).mean(0)
        else:
            raise ValueError('Empty s,t arrays (empty batches)')