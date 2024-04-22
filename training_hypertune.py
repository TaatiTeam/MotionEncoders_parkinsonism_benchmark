import os

import torch
import wandb
from tqdm import tqdm
import numpy as np

from learning.criterion import choose_criterion
from learning.optimizer import choose_optimizer, choose_scheduler
from learning.utils import AverageMeter, accuracy, save_checkpoint, assert_learning_params
from const import path, const
from utility.utils import is_substring, check_uniformity_and_get_first_elements

from collections import Counter, defaultdict
from sklearn.metrics import f1_score

import time
from collections import Counter

device = const._DEVICE

def final_test(model, test_loader, params):
    model.eval()
    video_logits = defaultdict(list)
    video_predclasses = defaultdict(list)
    video_labels = defaultdict(list)
    video_indices = defaultdict(list)
    video_states = defaultdict(list)
    video_names = defaultdict(list)

    loop = tqdm(test_loader)
    with torch.no_grad():        
        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device)
            metadata = metadata.to(device)
            if params['medication']:
                vi = video_idx.tolist()
                vn = [test_loader.dataset.video_names[i] for i in vi]
                on_off = [1 if 'on' in name else 0 for name in vn]
                on_off = torch.tensor(on_off, dtype=torch.float32, device=device)
                out = model(x, metadata, on_off)
            else:
                out = model(x, metadata)

            # Assuming out is a single tensor representing the output of the model for all clips
            summed_logits = torch.sum(out, dim=0).cpu().numpy()

            # Get the predicted class
            predicted_class = torch.argmax(torch.sum(out, dim=0).cpu()).item()
            
            # Append the logits, predicted class, and ground truth label for the video
            video_logits[video_idx.item()].append(summed_logits)
            video_predclasses[video_idx.item()].append(predicted_class)
            video_labels[video_idx.item()].append(y[0].item())
            video_indices[video_idx.item()].append(video_idx)
            
            # Retrieve and store the video name using video_idx
            video_name = test_loader.dataset.video_names[video_idx.item()]
            # video_names.append(video_name)
            video_names[video_idx.item()].append(video_name)
            
            if is_substring('on', test_loader.dataset.video_names[video_idx]):
                video_states[video_idx.item()].append('ON')
            else:
                video_states[video_idx.item()].append('OFF')
            
    #Just to make sure everything is ok with the process of gathering clips
    video_labels = check_uniformity_and_get_first_elements(list(video_labels.values()))
    video_indices = check_uniformity_and_get_first_elements(list(video_indices.values()))
    video_states = check_uniformity_and_get_first_elements(list(video_states.values()))
    video_names = check_uniformity_and_get_first_elements(list(video_names.values()))
    
    # Summing logits in each clip
    summed_video_logits = {idx: np.sum(logits, axis=0) for idx, logits in video_logits.items()}
    # Majority vote for predicted classes
    majority_vote_classes = {}
    for idx, classes in video_predclasses.items():
        class_counts = Counter(classes)
        majority_class = class_counts.most_common(1)[0][0]
        majority_vote_classes[idx] = majority_class
                
    return list(majority_vote_classes.values()), video_labels, list(summed_video_logits.values()), video_states, video_names


def validate_model(model, validation_loader, params, class_weights):
    criterion = choose_criterion(params['criterion'], params, class_weights)

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
    else:
        raise Exception("Cuda is not enabled")
    
    model.eval()
    accuracies = AverageMeter()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        video_predictions = defaultdict(list)
        video_predictions_labels = defaultdict(list)
        video_labels = {}

        for x, y, video_idx, metadata in validation_loader:
            x, y = x.to(device), y.to(device)
            metadata = metadata.to(device)
            batch_size = x.shape[0]

            if params['medication']:
                vi = video_idx.tolist()
                vn = [validation_loader.dataset.video_names[i] for i in vi]
                on_off = [1 if 'on' in name else 0 for name in vn]
                on_off = torch.tensor(on_off, dtype=torch.float32, device=device)
                out = model(x, metadata, on_off)
            else:
                out = model(x, metadata)
            _, out_label = torch.max(out, 1)

            loss = criterion(out, y)
            losses.update(loss.item(), batch_size)

            for i, idx in enumerate(video_idx):
                video_predictions_labels[idx.item()].append(out_label[i].detach())
                video_predictions[idx.item()].append(out[i].detach())
                video_labels[idx.item()] = y[i].item()

        total_correct = 0
        total_videos = 0
        for video_idx in video_predictions:
            predictions = video_predictions[video_idx]
            label_predictions = video_predictions_labels[video_idx]
            label_predictions = [label.item() for label in label_predictions]

            video_prediction = torch.stack(predictions).mean(dim=0).unsqueeze(0)
            video_label = torch.tensor([video_labels[video_idx]], device=video_prediction.device)
            label_counts = Counter(label_predictions)
            video_prediction_label, _ = label_counts.most_common(1)[0]

            acc, = accuracy(video_prediction, video_label)
            total_correct += acc

            total_videos += 1
            all_preds.append(video_prediction_label)
            all_labels.extend(video_label.cpu().numpy())

        video_accuracy = total_correct / total_videos
        accuracies.update(video_accuracy, total_videos)
        val_f1_score = f1_score(all_labels, all_preds, average='weighted')

        return losses.avg, accuracies.avg, val_f1_score


def train_model(params, class_weights, train_loader, val_loader, model, fold, backbone_name, mode="RUN"):
    assert_learning_params(params)
    
    criterion = choose_criterion(params['criterion'], params, class_weights)

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)
    else:
        raise Exception("Cuda is not enabled")

    optimizer = choose_optimizer(model, params)
    scheduler = choose_scheduler(optimizer, params)
    
    checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'],'models')
    if not os.path.exists(checkpoint_root_path): os.mkdir(checkpoint_root_path)

    loop = tqdm(range(params['epochs']), desc=f'Training (fold{fold})', unit="epoch")
    for epoch in loop:
        # print(f"[INFO] epoch {epoch}")
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        
        model.train()

        video_predictions = defaultdict(list)
        video_labels = {}
        
        epoch_start_time = time.time()
        for x, y, video_idx, metadata in train_loader:
            x, y = x.to(device), y.to(device)
            metadata = metadata.to(device)

            
            batch_size = x.shape[0]
            optimizer.zero_grad()

            if params['medication']:
                vi = video_idx.tolist()
                vn = [train_loader.dataset.video_names[i] for i in vi]
                on_off = [1 if 'on' in name else 0 for name in vn]
                on_off = torch.tensor(on_off, dtype=torch.float32, device=device)
                out = model(x, metadata, on_off)
            else:
                out = model(x, metadata)

            loss = criterion(out, y)
            train_loss.update(loss.item(), batch_size)

            for i, idx in enumerate(video_idx):
                video_predictions[idx.item()].append(out[i].detach())
                video_labels[idx.item()] = y[i].item()

            if params['lambda_l1'] > 0:
                learnable_params = torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])
                l1_regularization = torch.norm(learnable_params, p=1)
                
                loss += params['lambda_l1'] * l1_regularization

            loss.backward()
            optimizer.step()

            
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        # Compute accuracy per video
        total_correct = 0
        total_videos = 0
        for video_idx, predictions in video_predictions.items():
            video_prediction = torch.stack(predictions).mean(dim=0).unsqueeze(0)
            video_label = torch.tensor([video_labels[video_idx]], device=video_prediction.device)

            acc, = accuracy(video_prediction, video_label)
            total_correct += acc
            total_videos += 1

        video_accuracy = total_correct / total_videos
        train_acc.update(video_accuracy, total_videos)
        
        val_loss, val_acc, val_f1_score = validate_model(model, val_loader, params, class_weights)
        
        lr_backbone = optimizer.param_groups[0]['lr']
        
        if scheduler:
            scheduler.step()

        loop.set_postfix(train_loss=train_loss.avg, train_accuracy=train_acc.avg, 
                         val_loss=val_loss, val_accuracy=val_acc, val_f1_score=val_f1_score)

        log_wandb(epoch, fold, lr_backbone, train_acc, train_loss, 1, val_acc,
                val_loss, val_f1_score)
        
    if mode == "RUN":
        save_checkpoint(checkpoint_root_path, epoch, lr_backbone, optimizer, model, None, fold, latest=True)
        print(f'Checkpoint saved at: {checkpoint_root_path}')
    
    

def log_wandb(epoch, fold, lr_backbone, train_acc, train_loss, use_validation, validation_acc,
              validation_loss, validation_f1):
    log_dict = {
        f'train/fold{fold}_lr': lr_backbone,
        f'train_loss/fold{fold}_loss': train_loss.avg,
        f'train_accuracy/fold{fold}_accuracy': train_acc.avg,
        f'epoch': epoch,
    }
    if use_validation:
        log_dict[f'eval_loss/fold{fold}_loss'] = validation_loss
        log_dict[f'eval_acc/fold{fold}_accuracy'] = validation_acc
        log_dict[f'eval_f1/fold{fold}_f1'] = validation_f1
    wandb.log(log_dict)