import os
import sys

from colorama import Fore

from const import path


_NSEEDS = 8
import json


def generate_config(param, f_name):
    data_params = {
        'data_dim': 3,
        'in_data_dim': 3,
        'data_type': 'PD',
        'data_centered': False,
        'merge_last_dim': True,
        'use_validation': True,
        'simulate_confidence_score': False,
        'pretrained_dataset_name': 'NTU',
        'voting': False,
        'model_prefix': 'POTR_',
        'data_norm': 'unnorm',  # [minmax, unnorm, zscore]
        'source_seq_len': 80,
        'interpolate': True,
        "select_middle": False,
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'mirror_prob': 0.5,
        'noise_prob': 0.5,
        'axis_mask_prob': 0.5,
        'translation_frac': 0.05,
        'augmentation': []
    }

    model_params = {
        'model_dim': 128,
        'num_encoder_layers': 4,
        'num_heads': 4,
        'dim_ffn': 2048,
        'init_fn': 'xavier_init',
        'pose_embedding_type': 'gcn_enc',
        'pos_enc_alpha': 10,
        'pos_enc_beta': 500,
        'downstream_strategy': 'both',  # ['both', 'class', 'both_then_class'],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/poseforemer/pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt",
        'pose_format':  None,
        'classifier_dropout': 0.5,
        'classifier_hidden_dim': 2048,
        'preclass_rem_T': True
    }

    learning_params = {
        'wandb_name': 'poseforemer',
        'batch_size': 256,
        'lr_backbone': 0.0001,
        'lr_head': 0.001,
        'epochs': 100,
        'steps_per_epoch': 200,
        'dropout': 0.3,
        'max_gradient_norm': 0.1,
        'lr_step_size': 1,
        'learning_rate_fn': 'step',
        'criterion': 'WCELoss',
        'warmup_epochs': 10,
        'smoothing_scale': 0.1,
        'optimizer': 'AdamW',
        'stopping_tolerance': 10,
        'weight_decay': 0.00001,
        'lr_decay': 0.99,
        'experiment_name': '',
    }

    params = {**param, **data_params, **model_params, **learning_params}

    f = open("./configs/poseformer/" + f_name, "rb")
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys():
            print("Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
            print(Fore.RED + 'Configuration mismatch at:' + p)
            sys.exit(1)
        params[p] = new_param[p]
        

    if params['dataset'] == 'PD':
        params['labels_path'] = path.PD_PATH_LABELS
        params['data_path'] = path.PD_PATH_POSES

    params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0]
    return params, new_param
