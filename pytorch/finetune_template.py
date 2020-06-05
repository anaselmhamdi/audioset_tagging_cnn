import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import main
import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, Sampler, BalancedSampler, 
    BalancedMixupSampler, EvaluateSampler, Collator)
from main import get_train_sampler
from evaluate import Evaluator
from utilities import get_filename
from models import *
from losses import get_loss_func
from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer)
import config
import traceback
import gc

class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

class Transfer_Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14_DecisionLevelAtt, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14_DecisionLevelAtt(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(pretrained_checkpoint_path)
        else:
            checkpoint = torch.load(pretrained_checkpoint_path, map_location='gpu')
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        clipwise_output = output_dict['clipwise_output']
        framewise_output = torch.log_softmax(self.fc_transfer(x), dim=-1)
        output_dict['framewise_output'] = framewise_output
        return output_dict

def train(args):

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    freeze_base = True
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    sample_rate = config.sample_rate
    classes_num = config.classes_num
    pretrain = True if pretrained_checkpoint_path else False
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)

    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    print('Load pretrained model successfully!')
    ###############Copying main.py####################
    workspace_input = args.workspace_input
    workspace_output = args.workspace_output
    data_type = 'balanced_train'
    loss_type = 'clip_bce'
    balanced = 'balanced'
    augmentation = 'none'
    batch_size = 1
    learning_rate = 1e-3
    resume_iteration = 0
    early_stop = 100000
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    num_workers = 8
    clip_samples = config.clip_samples
    loss_func = get_loss_func(loss_type)
    black_list_csv = 'metadata/black_list/groundtruth_weak_label_evaluation_set.csv'
    previous_loss = None

    train_indexes_hdf5_path = os.path.join(workspace_input, 'hdf5s', 'indexes', 
        '{}.h5'.format(data_type))

    eval_bal_indexes_hdf5_path = os.path.join(workspace_input, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace_input, 'hdf5s', 'indexes', 
        'eval.h5')

    checkpoints_dir = os.path.join(workspace_output, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace_output, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace_output, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num,freeze_base=freeze_base)
    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(clip_samples=clip_samples, classes_num=classes_num)

    # Train sampler
    (train_sampler, train_collector) = get_train_sampler(balanced, augmentation,
        workspace_input + 'hdf5s/indexes/balanced_train.h5', black_list_csv, batch_size)
    
    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=workspace_input + 'hdf5s/indexes/balanced_train.h5', batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=workspace_input + 'hdf5s/indexes/eval.h5', batch_size=batch_size)

    eval_collector = Collator(mixup_alpha=None)
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=train_collector, 
        num_workers=num_workers, pin_memory=True)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=eval_collector, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=eval_collector, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    bal_evaluator = Evaluator(model=model, generator=eval_bal_loader)
    test_evaluator = Evaluator(model=model, generator=eval_test_loader)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace_input, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(resume_checkpoint_path)
        else:
            checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    time1 = time.time()
    
    for iterate_n, batch_data_dict in enumerate(train_loader):
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Evaluate
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = bal_evaluator.evaluate()
            test_statistics = test_evaluator.evaluate()
                            
            logging.info('Validate bal mAP: {:.3f}'.format(
                np.mean(bal_statistics['average_precision'])))

            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 20000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()
        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            
            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""
        loss = loss_func(batch_output_dict, batch_target_dict)
        # Loss
        # try:
        #     loss = loss_func(batch_output_dict, batch_target_dict)
        # except:
        #     tensor = batch_output_dict['clipwise_output'].detach().numpy()
        #     arr = -1. * np.where(tensor > 0,0.,tensor)
        #     batch_output_dict['clipwise_output'] = torch.tensor(np.where(arr > 1,1.,arr),requires_grad=True)
        #     loss = loss_func(batch_output_dict, batch_target_dict)
        # Backward
        loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                .format(iteration, time.time() - time1))
            time1 = time.time()
        
        iteration += 1

        # Stop learning
        if iteration == early_stop:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace_input', type=str, required=True, default='')
    parser_train.add_argument('--workspace_output', type=str, required=True, default='')
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', default=True, type=bool)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')