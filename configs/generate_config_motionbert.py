import json
import sys

from colorama import Fore

from const import path


def generate_config(param, f_name):
    data_params = {
        'data_type': 'PD',  # options: "Kinect", "GastNet", "PCT", "ViTPose, "PD"
        'data_dim': 3,
        'in_data_dim': 2,
        'data_centered': True,
        'merge_last_dim': False,
        'use_validation': True,
        'simulate_confidence_score': True,
        'pretrained_dataset_name': 'h36m',
        'model_prefix': 'motionbert_',
        # options: mirror_reflection, random_rotation, random_translation
        # 'augmentation': [],
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'mirror_prob': 0.5,
        'noise_prob': 0.5,
        'axis_mask_prob': 0.5,
        'translation_frac': 0.05,
        'data_norm': "rescaling",
        'select_middle': True,
        'exclude_non_rgb_sequences': False
    }
    model_params = {
        'source_seq_len': 243,
        'num_joints': 17,
        'dim_feat': 512, #MotionBERT 512, Lite 256
        'dim_rep': 512,
        'depth': 5,
        'num_heads': 8,
        'mlp_ratio': 2, #MotionBERT 2, Lite 4
        'maxlen': 243,
        'classifier_dropout': 0.5,
        'merge_joints': False,
        'classifier_hidden_dims': [2048],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/motionbert/motionbert.bin"
    }
    learning_params = {
        'wandb_name': 'motionbert',
        'experiment_name': '',
        'batch_size': 256,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'AdamW',
        'lr_backbone': 0.0001,
        'lr_head': 0.001,
        'weight_decay': 0.01,
        'lambda_l1': 0.0,
        'scheduler': "StepLR",
        'lr_decay': 0.99,
        'epochs': 20,
        'stopping_tolerance': 10,
        'lr_step_size': 1
    }

    params = {**param, **data_params, **model_params, **learning_params}

    f = open("./configs/motionbert/" + f_name, "rb")
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys():
            raise ValueError(
                "Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
        params[p] = new_param[p]



    params['labels_path'] = params['data_path']  # Data Path is the path to csv files by default


    if params['data_type'] == "PD":
        print("path.PD_PATH_LABELS", path.PD_PATH_LABELS)
        print("path.PD_PATH_POSES" , path.PD_PATH_POSES)
        params['labels_path'] = path.PD_PATH_LABELS
        params['data_path'] = path.PD_PATH_POSES


    params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0]
    return params, new_param
