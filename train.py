# load data, load model, checkpoint, save weights, logger

import os
import time
import csv
import yaml
import torch
import cv2
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim
from datetime import timedelta
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from dataloader.nyu_loader import NYUDataset
from experiment import DPT
from util.gpu_config import get_batch_size



def train(config):
    print("Initialize")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    print("Show available devices: ", torch.cuda.device_count())


    batch_size = config['experiment']['batch_size']
    epochs = config['experiment']['epochs']
    lr = config['experiment']['learning_rate']
    logs_path = config['experiment']['logs_path']

    print(f'BATCHSIZE: {batch_size}')
    print(f'EPOCHS: {epochs}')
    print('#############################')

    start = time.time()

    #data parallelization
    batch_size = get_batch_size(config)
    #load model
    if config['model']['model_type']=='dpt_hybrid_nyu':
        model = DPT(config)

    # logging setup
    logger = TensorBoardLogger(logs_path, 
                               name='finetune',
                               log_graph=True)

    train_dataset = NYUDataset(config['dataset']['data_path'], type='train')
    val_dataset = NYUDataset(config['dataset']['data_path'], type='val')

    train_loader = DataLoader(train_dataset, 
                              batch_size=config['experiment']['batch_size'], 
                              shuffle=True,
                            #   prefetch_factor=10, # increase or decrease based on free gpu mem
                            #   pin_memory=True,
                              num_workers=4*config['experiment']['gpus'] if torch.cuda.is_available() else 0)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config['experiment']['batch_size'],
                            # prefetch_factor=10, # increase or decrease based on free gpu mem
                            # pin_memory=True,
                            num_workers=4*config['experiment']['gpus'] if torch.cuda.is_available() else 0)

    # checkpointing
    # model_ckpt = ModelCheckpoint(every_n_epochs=5,
    #                              save_top_k=-1,
    #                              filename='dpt-finetune-{epoch}')
    
    # # save s,t weights
    # st_ckpt = TensorCheckpoint(every_n_epochs=5)

    print("Dataset loaded****")


    if torch.cuda.is_available():
        if config['experiment']['gpus'] > 1:
            if(config['model']['load_ckpt']):
                path =config['model']['model_path']
            else:
                path=None
            trainer = pl.Trainer(resume_from_checkpoint=path,
                                gpus=config['experiment']['gpus'], 
                                max_epochs=model.epochs,
                                accelerator= 'ddp',
                                logger=logger,
                                num_sanity_val_steps=0,
                                progress_bar_refresh_rate=None if config['experiment']['verbose'] else 0)
        else:
            if(config['model']['load_ckpt']):
                path =config['model']['model_path']
            else:
                path=None
            trainer = pl.Trainer(resume_from_checkpoint=path,
                                 gpus=config['experiment']['gpus'],
                                 max_epochs=model.epochs,
                                 logger=logger,
                                 num_sanity_val_steps=0,
                                 progress_bar_refresh_rate=None if config['experiment']['verbose'] else 0)
    else:
        trainer = pl.Trainer(max_epochs=1, logger=logger)
        
    print('Training')

    try:
        cv2.setNumThreads(0) # disable cv2 threading to avoid deadlocks
        
        start = time.time()
        trainer.fit(model, 
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

    except Exception as e:
        print('Training was halted due to the following error:')
        raise

    else:
        print(f'Training completed in {timedelta(seconds=round(time.time()-start,2))}')

    finally:
        exp_idx = len(list(filter(lambda f: '.pt' in f, os.listdir(os.path.join(logs_path)))))
        print(f'Training checkpoints and logs are saved in {trainer.log_dir}')
        print(f'Final trained weights saved in finetune{exp_idx}.pt')
        torch.save(model.state_dict(), os.path.join(logs_path, f'finetune{exp_idx}.pt'))

    logger.save()
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generic runner model')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config/train.yaml')
                        
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        
    # fix the seed for reproducibility
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    #set torch options
    cudnn.benchmark = True #better runtime performance if input image size do not change
    cudnn.enabled = True

    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


    # if not os.path.isdir('checkpoints'):
    #     os.mkdir('checkpoints')

    train(config)