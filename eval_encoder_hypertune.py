import sys
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import wandb

from configs import generate_config_poseformer, generate_config_motionbert, generate_config_poseformerv2, generate_config_mixste, generate_config_motionagformer

from data.dataloaders import *
from const import path
from utility.utils import set_random_seed
from test import *

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + "/../")


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_results(rep, confusion, rep_name, conf_name, out_p):
    print(rep)
    fig, ax = plt.subplots(figsize=(10, 8)) 
    sns.heatmap(confusion, annot=True, ax=ax, cmap="Blues", fmt='g', annot_kws={"size": 26})
    ax.set_xlabel('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    ax.set_title('Confusion Matrix', fontsize=30)
    ax.xaxis.set_ticklabels(['class 0', 'class 1', 'class 2'], fontsize=22)  # Modify class names as needed
    ax.yaxis.set_ticklabels(['class 0', 'class 1', 'class 2'], fontsize=22)
    # Save the figure
    plt.savefig(os.path.join(out_p, conf_name))
    plt.close(fig)
    with open(os.path.join(out_p, rep_name), "w") as text_file:
        text_file.write(rep)
    
    artifact = wandb.Artifact(f'confusion_matrices', type='image-results')
    artifact.add_file(os.path.join(out_p, conf_name))
    wandb.log_artifact(artifact)
    
    artifact = wandb.Artifact('reports', type='txtfile-results')
    artifact.add_file(os.path.join(out_p, rep_name))
    wandb.log_artifact(artifact)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( poseformer, ''motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only', help='train mode( end2end, classifier_only )')
    parser.add_argument('--dataset', type=str, default='PD',help='**currently code only works for PD')
    parser.add_argument('--data_path', type=str,default=path.PD_PATH_POSES)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int, help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--last_run_foldnum', default='7', type=str)
    parser.add_argument('--readstudyfrom', default=1, type=int)
    
    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str, help="add metadata prob to the training 'gender,age,bmi,height,weight'")

    args = parser.parse_args()

    param = vars(args)
    
    param['metadata'] = param['metadata'].split(',') if param['metadata'] else []

    torch.backends.cudnn.benchmark = False
    
    backbone_name = param['backbone']

    # TODO: Make it scalable
    if backbone_name == 'poseformer':
        conf_path = './configs/poseformer/'
    elif backbone_name == 'motionbert':
        conf_path = './configs/motionbert/'
    elif backbone_name == 'poseformerv2':
        conf_path = './configs/poseformerv2'
    elif backbone_name == 'mixste':
        conf_path = './configs/mixste'
    elif backbone_name == 'motionagformer':
        conf_path = './configs/motionagformer'
    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")

    for fi in sorted(os.listdir(conf_path)):

        if backbone_name == 'poseformer':
            params, new_params = generate_config_poseformer.generate_config(param, fi)
        elif backbone_name == 'motionbert':
            params, new_params = generate_config_motionbert.generate_config(param, fi)
        elif backbone_name == 'poseformerv2':
            params, new_params = generate_config_poseformerv2.generate_config(param, fi)
        elif backbone_name == 'mixste':
            params, new_params = generate_config_mixste.generate_config(param, fi)
        elif backbone_name == 'motionagformer':
            params, new_params = generate_config_motionagformer.generate_config(param, fi)
        else:
            raise NotImplementedError(f"Backbone '{param['backbone']}' does not exist.")

        if param['dataset'] == 'PD':
            num_folds = 23
            params['num_classes'] = 3  
        else:
            raise NotImplementedError(f"dataset '{param['dataset']}' is not supported.")

        all_folds = range(1, num_folds + 1)
        set_random_seed(param['seed'])

        test_and_report(params, new_params, all_folds, backbone_name, _DEVICE)
            