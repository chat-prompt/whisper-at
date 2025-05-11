# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
os.environ['TRANSFORMERS_CACHE'] = './tr/'
import ast
import pickle
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader_feat as dataloader
import numpy as np
from traintest import train, validate
from models import TLTR

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE", "SONY_BCE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--model_size", type=str, default='medium.en', help="The model size")
parser.add_argument("--label_smooth", type=float, default=0.0, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--ftmode", type=str, default='last', help="pretrained model path")
parser.add_argument("--pretrain_epoch", type=int, default=0, help="number of pretrained epochs")
parser.add_argument("--head_lr", type=float, default=1.0, help="learning rate ratio between mlp/base")
parser.add_argument("--pretrained_model", type=str, default=None, help="path to pretrained model")
parser.add_argument("--weight_decay", type=float, default=5e-7, help="weight decay")
parser.add_argument("--freeze_original_classes", action="store_true", help="Freeze weights for original 527 AudioSet classes and only train new SONYC classes")
args = parser.parse_args()

val_tar_path = None
eval_tar_path = None

if args.dataset == 'esc':
    if args.model_size == 'hubert-xlarge-ls960-ft' or args.model_size == 'wav2vec2-large-robust-ft-swbd-300h':
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/' + args.model_size
    else:
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/whisper_' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_esc_pool/whisper_' + args.model_size
    shuffle = True
elif args.dataset == 'as-bal' or args.dataset == 'as-full':
    if args.model_size == 'hubert-xlarge-ls960-ft' or args.model_size == 'wav2vec2-large-robust-ft-swbd-300h':
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/' + args.model_size
    else:
        train_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_full/whisper_' + args.model_size
        eval_tar_path = '/data/sls/scratch/yuangong/whisper-a/feat_as_eval/whisper_' + args.model_size
    shuffle = True
elif args.dataset == 'sonyc':
    train_tar_path = '/mnt/ssd_disk/datasets/sonyc-ust/features/train'
    # val_tar_path = '/mnt/ssd_disk/datasets/sonyc-ust/features/val'
    # eval_tar_path = '/mnt/ssd_disk/datasets/sonyc-ust/features/test'
    eval_tar_path = '/mnt/ssd_disk/datasets/audioset_sonyc_combined/features/val'
    shuffle = True
elif args.dataset == 'audioset_sonyc':
    train_tar_path = '/mnt/ssd_disk/datasets/audioset_sonyc_combined/features/train'
    eval_tar_path = '/mnt/ssd_disk/datasets/audioset_sonyc_combined/features/val'
    shuffle = True

audio_conf = {'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'label_smooth': args.label_smooth, 'tar_path': train_tar_path}
val_audio_conf = None
eval_audio_conf = None

if val_tar_path is not None and eval_tar_path is not None:
    val_audio_conf = {'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'tar_path': val_tar_path}
    eval_audio_conf = {'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'tar_path': eval_tar_path}
elif val_tar_path is None and eval_tar_path is not None:
    val_audio_conf = {'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'tar_path': eval_tar_path}

if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=eval_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

def get_feat_shape(path, args):
    mdl_size = args.model_size
    n_rep_dim_dict = {'tiny.en': 384, 'tiny': 384, 'base.en': 512, 'base': 512, 'small.en': 768, 'small': 768, 'medium.en': 1024, 'medium': 1024, 'large-v1': 1280, 'large-v2': 1280, 'wav2vec2-large-robust-ft-swbd-300h': 1024, 'hubert-xlarge-ls960-ft': 1280}
    n_layer_dict = {'tiny.en': 4, 'tiny': 4, 'base.en': 6, 'base': 6, 'small.en': 12, 'small': 12, 'medium.en': 24, 'medium': 24, 'large-v1': 32, 'large-v2': 32, 'wav2vec2-large-robust-ft-swbd-300h': 24, 'hubert-xlarge-ls960-ft': 48}
    return n_layer_dict[mdl_size], n_rep_dim_dict[mdl_size]

if 'whisper-high' in args.model:
    mode = args.model.split('-')[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layer, rep_dim = get_feat_shape(train_tar_path, args)
    print(mode, args.model_size, n_layer, rep_dim)
    audio_model = TLTR(label_dim=args.n_class, n_layer=n_layer, rep_dim=rep_dim, mode=mode)
else:
    raise ValueError('model not supported')

print(audio_model)

if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
    print(f"Loading pretrained model from {args.pretrained_model}")
    
    # 1. 사전 학습된 모델의 state_dict 로드
    raw_pretrained_state = torch.load(args.pretrained_model, map_location=device)
    
    # 2. 현재 모델의 state_dict 가져오기 (나중에 로드할 때 사용)
    current_model_state_for_loading = audio_model.state_dict()
    # 최종적으로 audio_model에 로드될 state_dict를 준비 (현재 모델 상태로 초기화)
    final_state_to_load = current_model_state_for_loading.copy()

    # 3. 키 정리를 위한 헬퍼 함수
    def get_clean_key(key_str):
        if key_str.startswith('module.'):
            key_str = key_str[7:]
        if key_str.startswith('at_model.'): # whisper-at 평가 스크립트에서 발견된 접두사
            key_str = key_str[9:]
        return key_str

    # 4. 사전 학습된 state_dict의 키를 정리하여 조회용으로 만듦
    pretrained_state_for_lookup = {get_clean_key(k): v for k, v in raw_pretrained_state.items()}

    # 5. 분류 레이어의 기본 이름 정의 (접두사 없음)
    classifier_base_names = ['mlp_layer.1.weight', 'mlp_layer.1.bias']
    
    print("Attempting to load weights...")
    loaded_count = 0
    adapted_count = 0

    # 6. 현재 모델의 각 레이어에 대해 가중치 로드 시도
    for model_key_original in final_state_to_load.keys():
        # 현재 모델 키에서 접두사를 제거하여 기본 키 이름 획득
        # (audio_model이 DataParallel로 래핑된 경우 'module.' 접두사가 있을 수 있음)
        clean_model_key = get_clean_key(model_key_original) 

        if clean_model_key in pretrained_state_for_lookup:
            pretrained_tensor = pretrained_state_for_lookup[clean_model_key]
            current_tensor_template = final_state_to_load[model_key_original]

            # A. 분류 레이어 처리 (크기 조정 가능성 있음)
            if clean_model_key in classifier_base_names:
                if pretrained_tensor.size(0) < current_tensor_template.size(0): # 클래스 수 증가
                    print(f"Adapting classifier layer: {model_key_original} (base: {clean_model_key}) "
                          f"from {pretrained_tensor.size()} to {current_tensor_template.size()}")
                    
                    # 기존 클래스 가중치 복사
                    current_tensor_template[:pretrained_tensor.size(0)] = pretrained_tensor.narrow(0, 0, pretrained_tensor.size(0)) if 'weight' in clean_model_key else pretrained_tensor.narrow(0, 0, pretrained_tensor.size(0))


                    # 새 클래스 가중치 초기화
                    if 'weight' in clean_model_key:
                         # weight의 경우 [out_features, in_features]
                        new_part = current_tensor_template[pretrained_tensor.size(0):, :]
                    else: # bias의 경우 [out_features]
                        new_part = current_tensor_template[pretrained_tensor.size(0):]

                    mean = pretrained_tensor.mean().item()
                    std = pretrained_tensor.std().item()
                    if std < 1e-6: # std가 너무 작으면 초기화 시 문제 발생 가능
                        print(f"Warning: std of pretrained tensor for {clean_model_key} is very small ({std}). Using 0.01 for new part initialization.")
                        std = 0.01 # 기본 std 값 사용
                    
                    torch.nn.init.normal_(new_part, mean=mean, std=std)
                    # final_state_to_load[model_key_original]은 이미 current_tensor_template을 가리키므로,
                    # current_tensor_template의 수정이 반영됨.
                    adapted_count +=1
                elif pretrained_tensor.size() == current_tensor_template.size():
                    final_state_to_load[model_key_original] = pretrained_tensor
                    loaded_count += 1
                else: # 사전 학습 모델의 클래스 수가 더 많거나 다른 크기 불일치
                    print(f"Warning: Size mismatch for classifier layer {model_key_original} (base: {clean_model_key}). "
                          f"Pretrained: {pretrained_tensor.size()}, Model: {current_tensor_template.size()}. Skipping this layer.")
            
            # B. 분류 레이어가 아닌 다른 레이어 처리
            else:
                if pretrained_tensor.size() == current_tensor_template.size():
                    final_state_to_load[model_key_original] = pretrained_tensor
                    loaded_count += 1
                else:
                    print(f"Warning: Size mismatch for non-classifier layer {model_key_original} (base: {clean_model_key}). "
                          f"Pretrained: {pretrained_tensor.size()}, Model: {current_tensor_template.size()}. Skipping this layer.")
        else:
            print(f"Warning: Key {clean_model_key} (original: {model_key_original}) not found in cleaned pretrained_state. Skipping this layer.")

    # 7. 최종적으로 준비된 state_dict를 현재 모델에 로드
    #    strict=False는 final_state_to_load에 있지만 audio_model에 없는 키, 
    #    또는 그 반대의 경우를 허용. 현재 로직상 final_state_to_load는 audio_model의 모든 키를 가지므로,
    #    주로 크기가 다른 레이어가 얼마나 있었는지 등을 나타냄.
    load_status = audio_model.load_state_dict(final_state_to_load, strict=False)
    print(f"Weight loading status - Missing keys: {load_status.missing_keys}, Unexpected keys: {load_status.unexpected_keys}")
    print(f"Successfully loaded {loaded_count} layers and adapted {adapted_count} classifier layers.")
    print("Pretrained weights loading process finished.")

    # 모델 가중치 로딩 후, 기존 클래스 가중치 고정 코드 추가
    if args.pretrained_model is not None and args.freeze_original_classes:
        print("Freezing original AudioSet class weights...")
        for name, param in audio_model.named_parameters():
            if 'mlp_layer.1.weight' in name:
                # 기존 클래스(0-526)에 해당하는 가중치 행 고정
                param.data[:527, :].requires_grad = False
            elif 'mlp_layer.1.bias' in name:
                # 기존 클래스에 해당하는 바이어스 고정
                param.data[:527].requires_grad = False

# use data parallel
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

code_path = args.exp_dir + '/src/'
if os.path.exists(code_path) == False:
    os.mkdir(code_path)
copy_path = '/home/taemyung_heo/workspace/github/whisper-at/src/'
os.system('cp ' + copy_path + '/*.sh ' + code_path)
os.system('cp ' + copy_path + '/*.py ' + code_path)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)


def wa_model(exp_dir, start_epoch=16, end_epoch=30):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        if os.path.exists(exp_dir + '/models/audio_model.' + str(epoch) + '.pth') == True:
            sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
            for key in sdA:
                sdA[key] = sdA[key] + sdB[key]
            model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    torch.save(sdA, exp_dir + '/models/audio_model_wa.pth')
    return sdA

# do model weight averaging
if args.wa == True:
    sdA = wa_model(args.exp_dir, args.wa_start, args.wa_end)
    msg = audio_model.load_state_dict(sdA, strict=True)
    print(msg)
    audio_model.eval()
    stats, _ = validate(audio_model, val_loader, args)
    wa_res = np.mean([stat['AP'] for stat in stats])
    wa_res_sonyc = np.mean([stat['AP'] for stat in stats[527:]])
    print('val mAP of model with weights averaged from checkpoint {:d}-{:d} is {:.4f}, SONYC mAP: {:.4f}'.format(args.wa_start, args.wa_end, wa_res, wa_res_sonyc))
    np.savetxt(args.exp_dir + '/wa_res.csv', [args.wa_start, args.wa_end, wa_res, wa_res_sonyc], delimiter=',')

    if args.data_eval != None:
        stats, _ = validate(audio_model, eval_loader, args)
        wa_res = np.mean([stat['AP'] for stat in stats])
        wa_res_sonyc = np.mean([stat['AP'] for stat in stats[527:]])
        print('test mAP of model with weights averaged from checkpoint {:d}-{:d} is {:.4f}, SONYC mAP: {:.4f}'.format(args.wa_start, args.wa_end, wa_res, wa_res_sonyc))
        np.savetxt(args.exp_dir + '/wa_res_test.csv', [args.wa_start, args.wa_end, wa_res, wa_res_sonyc], delimiter=',')