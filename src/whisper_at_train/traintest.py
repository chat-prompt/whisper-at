# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pandas as pd
import json
import pickle
from torch.cuda.amp import autocast,GradScaler

def load_mid_to_index_map(label_csv_path):
    """class_labels_indices_extended.csv에서 MID -> 정수 인덱스 맵을 로드합니다."""
    try:
        df = pd.read_csv(label_csv_path)
        if 'mid' not in df.columns or 'index' not in df.columns:
            # 헤더가 없는 경우 시도 (index, mid, display_name 순서 가정)
            df = pd.read_csv(label_csv_path, header=None, names=['index', 'mid', 'display_name'])
            if not (isinstance(df['mid'].iloc[0], str) and df['mid'].iloc[0].startswith(('/m/', '/t/'))):
                 raise ValueError("Label CSV must contain 'mid' and 'index' columns.")
        mid_to_idx = pd.Series(df['index'].values, index=df['mid']).to_dict()
        return mid_to_idx
    except Exception as e:
        print(f"Error loading MID-to-index map from {label_csv_path}: {e}")
        raise

def calculate_pos_weights(data_json_path, label_csv_path, num_classes, sonyc_class_indices=None, sonyc_boost_factor=1.0, clip_min=None, clip_max=None):
    """
    학습 데이터셋의 클래스 빈도를 기반으로 pos_weight를 계산합니다.
    Args:
        data_json_path (str): 학습 데이터 JSON 파일 경로 (예: combined_train.json).
                               'labels' 필드는 쉼표로 구분된 MID 문자열을 포함해야 함.
        label_csv_path (str): MID를 정수 인덱스로 매핑하는 CSV 파일 경로.
        num_classes (int): 전체 클래스 수 (예: 533).
        sonyc_class_indices (list, optional): 추가 가중치를 부여할 SONYC 클래스 인덱스 리스트.
                                            예: list(range(527, 533))
        sonyc_boost_factor (float, optional): SONYC 클래스에 적용할 추가 가중치 배율.
    Returns:
        torch.Tensor: 각 클래스에 대한 pos_weight 텐서.
    """
    print(f"Calculating pos_weights from {data_json_path} with {num_classes} classes.")
    mid_to_idx = load_mid_to_index_map(label_csv_path)

    class_counts = np.zeros(num_classes, dtype=np.float32)
    total_valid_samples = 0 # 유효한 레이블을 가진 샘플 수

    with open(data_json_path, 'r') as f:
        dataset = json.load(f)

    for entry in dataset.get('data', []):
        labels_str = entry.get('labels', "")
        if not labels_str:
            continue

        has_valid_label_in_entry = False
        mids = [m.strip() for m in labels_str.split(',') if m.strip()]
        for mid in mids:
            if mid in mid_to_idx:
                class_idx = mid_to_idx[mid]
                if 0 <= class_idx < num_classes:
                    class_counts[class_idx] += 1
                    has_valid_label_in_entry = True
        if has_valid_label_in_entry:
            total_valid_samples +=1

    if total_valid_samples == 0:
        print("Warning: No valid samples found to calculate pos_weights. Returning default weights (all 1s).")
        return torch.ones(num_classes)

    # pos_weight 계산: (Negative 샘플 수) / (Positive 샘플 수)
    # 또는 (전체 샘플 수 - Positive 샘플 수) / Positive 샘플 수
    # 분모가 0이 되는 것을 방지하기 위해 작은 값(epsilon)을 더하거나, 최소 카운트를 설정할 수 있음.
    epsilon = 1e-6 # 0으로 나누는 것을 방지
    pos_weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        # 각 샘플은 여러 레이블을 가질 수 있으므로, total_valid_samples를 N으로 사용
        # class_counts[i]는 해당 클래스가 positive로 나타난 샘플 수
        # (total_valid_samples - class_counts[i])는 해당 클래스가 negative로 나타난 샘플 수 (근사치)
        if class_counts[i] > 0:
            pos_weights[i] = (total_valid_samples - class_counts[i]) / (class_counts[i] + epsilon)
        else:
            # 이 클래스가 학습 데이터에 전혀 등장하지 않은 경우, 가중치를 어떻게 설정할지 결정 필요.
            # 매우 큰 값을 주거나, 1로 설정하거나, 또는 학습에서 제외하는 방법도 있음.
            # 여기서는 일단 매우 큰 값 (모든 샘플이 negative라고 가정)
            pos_weights[i] = total_valid_samples / epsilon # 또는 적절한 큰 값 (예: 1000)

    # SONYC 클래스에 추가 가중치 적용
    if sonyc_class_indices and sonyc_boost_factor > 1.0:
        print(f"Boosting SONYC classes {sonyc_class_indices} by factor {sonyc_boost_factor}")
        for idx in sonyc_class_indices:
            if 0 <= idx < num_classes:
                pos_weights[idx] *= sonyc_boost_factor

    # 가중치가 너무 커지거나 작아지는 것을 방지하기 위해 클리핑 가능
    if clip_min is not None and clip_max is not None:
        pos_weights = np.clip(pos_weights, clip_min, clip_max)

    print(f"Calculated class counts (first 10): {class_counts[:10]}")
    print(f"Calculated pos_weights (first 10): {pos_weights[:10]}")
    if sonyc_class_indices:
         print(f"Calculated pos_weights for SONYC classes ({sonyc_class_indices}): {pos_weights[sonyc_class_indices[0]:sonyc_class_indices[-1]+1]}")

    return torch.from_numpy(pos_weights)


class SonyNewClassBCELoss(nn.Module):
    """신규 SONYC 클래스(527-532)에 대한 손실만 계산하는 손실 함수"""
    def __init__(self):
        super(SonyNewClassBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, predictions, targets):
        # 모든 클래스에 대한 BCE 손실 계산
        all_losses = self.bce_loss(predictions, targets)
        
        # 신규 SONYC 클래스에 대한 손실만 추출
        sonyc_losses = all_losses[:, 527:533]
        
        # SONYC 클래스 손실의 평균 계산
        reduced_loss = sonyc_losses.mean()
        
        return reduced_loss

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'BCE_WEIGHTED':
# SONYC 클래스 인덱스 (0-based): 527부터 532까지
        sonyc_indices = list(range(527, 533)) 
        # SONYC 클래스에 부여할 추가 가중치 배율 (예: 2.0, 3.0 등 실험 필요)
        sonyc_boost = 1.0 # 이 값을 셸 스크립트에서 인자로 받는 것도 좋음

        try:
            calculated_pos_weights = calculate_pos_weights(
                args.data_train, 
                args.label_csv, 
                args.n_class,
                sonyc_class_indices=sonyc_indices,
                sonyc_boost_factor=sonyc_boost,
                clip_min=0.1,
                clip_max=50.0
            ).to(device)
            print("Using calculated pos_weights for BCEWithLogitsLoss.")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=calculated_pos_weights)
        except Exception as e:
            print(f"Error calculating pos_weights: {e}. Using standard BCEWithLogitsLoss without pos_weight.")
            loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'SONY_BCE':
        loss_fn = SonyNewClassBCELoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, labels) in enumerate(train_loader):

            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                audio_output = audio_model(a_input)
                loss = loss_fn(audio_output, labels)

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

            # for audioset-full, break every 10% of the epoch, i.e., equivalent epochs = 0.1 * specified epochs
            if args.dataset == 'as-full':
                if i > 0.1 * len(train_loader):
                    break

        print('start validation')

        stats, valid_loss = validate(audio_model, test_loader, args)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            pass
            #torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (a_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device, non_blocking=True)

            with autocast():
                audio_output = audio_model(a_input)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # SONYC 클래스에 대한 별도 성능 평가 추가
        if hasattr(args, 'n_class') and args.n_class > 527:
            all_mAP = np.mean([stat['AP'] for stat in stats])
            sonyc_mAP = np.mean([stat['AP'] for stat in stats[527:args.n_class]])
            original_mAP = np.mean([stat['AP'] for stat in stats[:527]])
            
            print(f"All classes mAP: {all_mAP:.6f}")
            print(f"Original AudioSet classes mAP: {original_mAP:.6f}")
            print(f"SONYC classes mAP: {sonyc_mAP:.6f}")

        return stats, loss