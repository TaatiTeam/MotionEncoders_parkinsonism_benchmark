import os

import torch
import wandb

from collections import Counter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def upload_checkpoints_to_wandb(latest_epoch_path, best_epoch_path):
    artifact = wandb.Artifact(f'model', type='model')
    artifact.add_file(latest_epoch_path)
    artifact.add_file(best_epoch_path)
    wandb.log_artifact(artifact)


def save_checkpoint(checkpoint_root_path, epoch, lr, optimizer, model, best_accuracy, fold, latest):
    checkpoint_path_fold = os.path.join(checkpoint_root_path, f"fold{fold}")
    if not os.path.exists(checkpoint_path_fold):
        os.makedirs(checkpoint_path_fold)
    checkpoint_path = os.path.join(checkpoint_path_fold,
                                   'latest_epoch.pth.tr' if latest else 'best_epoch.pth.tr')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'best_accuracy': best_accuracy
    }, checkpoint_path)


def assert_learning_params(params):
    """Makes sure the learning parameters is set as parameters (To avoid raising error during training)"""
    learning_params = ['batch_size', 'criterion', 'optimizer', 'lr_backbone', 'lr_head', 'weight_decay', 'epochs',
                       'stopping_tolerance']
    for learning_param in learning_params:
        assert learning_param in params, f'"{learning_param}" is not set in params.'

def compute_class_weights(data_loader):
    class_counts = Counter()
    total_samples = 0
    num_classes = 0

    for _, targets, _, _ in data_loader:
        class_counts.update(targets.tolist())
        total_samples += len(targets)

    class_weights = []

    num_classes = len(class_counts)
    for i in range(num_classes):
        count = class_counts[i]
        weight = 0.0 if count == 0 else total_samples / (num_classes * count)
        class_weights.append(weight)
        
        total_weights = sum(class_weights)
        normalized_class_weights = [weight / total_weights for weight in class_weights]

    return normalized_class_weights
