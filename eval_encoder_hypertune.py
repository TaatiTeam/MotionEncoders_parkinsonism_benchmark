import sys
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn

import wandb

from configs import generate_config_poseformer, generate_config_motionbert, generate_config_poseformerv2, generate_config_mixste, generate_config_motionagformer

from data.dataloaders import *
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from training_hypertune import train_model, validate_model, final_test
from const import path
from utility.utils import set_random_seed
from test_hypertune import *
import joblib
import datetime

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
                

class SaveStudyCallback:
    def __init__(self, save_frequency, file_path):
        self.save_frequency = save_frequency
        self.file_path = file_path
        self.trial_counter = 0
        if not os.path.exists(self.file_path.split('study_mid.pkl')[0]):
            os.mkdir(self.file_path.split('study_mid.pkl')[0])

    def __call__(self, study, trial):
        self.trial_counter += 1
        if self.trial_counter % self.save_frequency == 0:
            joblib.dump(study, self.file_path)


def objective_LOSO(trial, splits, run_name, params):
    #SET approperiate classifier dim according to the baseline feature dim
    if backbone_name == 'poseformer':
        classifier_hidden_dims_choices = {   #POTR
        'option1': [],
        }
        ep = [5, 10, 20, 30, 50, 100]
    elif backbone_name == 'motionbert':
        classifier_hidden_dims_choices = {   #motionBert
            'option1': [],
        }
        ep = [5, 10, 20, 30, 50]
    elif backbone_name == 'poseformerv2':
        classifier_hidden_dims_choices = {'option1': []}
        ep = [5, 10, 20, 30, 50]
    elif backbone_name == 'mixste':
        classifier_hidden_dims_choices = {'option1': []}
        ep = [5, 10, 20, 30, 50]
    elif backbone_name == 'motionagformer':
        classifier_hidden_dims_choices = {'option1': []}
        ep = [5, 10, 20, 30, 50]
    classifier_op = list(classifier_hidden_dims_choices.keys())

    
    lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001, 0.00001, 0.000001])
    num_epochs = trial.suggest_categorical('num_epochs', ep) 
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'RMSprop', 'SGD'])
    lambda_l1 = trial.suggest_categorical('lambda_l1', [0]) #[0, 0.0001, 0.001]
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.0, 0.3])
    use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True])
    chosen_option  = trial.suggest_categorical('classifier_hidden_dims', classifier_op)
    classifier_hidden_dims = classifier_hidden_dims_choices[chosen_option]

    
    params['classifier_dropout'] = dropout_rate
    params['classifier_hidden_dims'] = classifier_hidden_dims
    params['optimizer'] = optimizer_name
    params['lr_head'] = lr
    params['epochs'] = num_epochs
    params['lambda_l1'] = lambda_l1
    if use_weighted_loss:
        params['criterion'] = 'WCELoss'
    else:
        params['criterion'] = 'CrossEntropyLoss'
        
    if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop']:
        params['weight_decay'] = trial.suggest_float('weight_decay', 0, 1e-2)

    if params['optimizer'] == 'SGD':
        params['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
        
    print("========================================================================================")
    print("========================================================================================")
    print(f"Trial {trial.number}, lr: {lr}, num_epochs: {num_epochs}")
    print(f"optimizer_name: {optimizer_name}, dropout_rate: {dropout_rate}")
    print(f"batch_size: {params['batch_size']}, use_weighted_loss: {use_weighted_loss}")
    print("========================================================================================")
    print("========================================================================================")
    
    wandb.init(project='MotionEncoderEvaluator_PD', 
            group=run_name+'-LOSO', 
            job_type='optunatrial', 
            name=run_name + '_trial:' +format(trial.number, '05d'),
            reinit=True)
    
    aggregated_val_f1_score = 0  # Aggregate validation F1 score across all folds
    aggregated_val_loss = 0  # Aggregate validation loss across all folds
    aggregated_val_composite_score = 0

    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits, start=1):
        start_time = datetime.datetime.now()
        params['input_dim'] = train_dataset_fn.dataset._pose_dim
        params['pose_dim'] = train_dataset_fn.dataset._pose_dim
        params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS


        model_backbone = load_pretrained_backbone(params, backbone_name)
        model = MotionEncoder(backbone=model_backbone,
                                params=params,
                                num_classes=params['num_classes'],
                                num_joints=params['num_joints'],
                                train_mode=params['train_mode'])
        model = model.to(_DEVICE)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            
        if fold == 1:
            model_params = count_parameters(model)
            print(f"[INFO] Model has {model_params} parameters.")

        train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name, mode="Hypertune")
        # Evaluate the model
        val_loss, val_acc, val_f1_score = validate_model(model, val_dataset_fn, params, class_weights)
        
        composite_score = val_f1_score - val_loss
        
        aggregated_val_composite_score += composite_score
        aggregated_val_f1_score += val_f1_score
        aggregated_val_loss += val_loss
        
        end_time = datetime.datetime.now()
        
        duration = end_time - start_time
        print(f"Fold {fold} run time:", duration)

    # Compute average performance across all folds
    avg_val_f1_score = aggregated_val_f1_score / len(splits)
    avg_val_loss = aggregated_val_loss / len(splits)
    avg_val_composite_score = aggregated_val_composite_score / len(splits)
    wandb.log({
            f'final/avg_val_composite_score': avg_val_composite_score,
            f'final/avg_val_f1_score': avg_val_f1_score,
            f'final/avg_val_loss': avg_val_loss
        })
    wandb.config.update(params)
    wandb.finish()
    
    return avg_val_composite_score
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( poseformer, ''motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only', help='train mode( end2end, classifier_only )')
    parser.add_argument('--dataset', type=str, default='PD',help='**currently code only works for PD')
    parser.add_argument('--data_path', type=str,default=path.PD_PATH_POSES)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int, help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--ntrials', default=30, type=int, help='number of hyper-param tuning trials')
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

        test__hypertune(params, new_params, all_folds, backbone_name, _DEVICE)
            