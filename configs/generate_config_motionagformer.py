import json
import sys

from colorama import Fore

from const import path


def generate_config(param, f_name):
    data_params = {
        'data_type': 'Kinect',  # options: "Kinect", "GastNet", "PCT", "ViTPose"
        'data_dim': 3,
        'in_data_dim': 2,
        'data_centered': True,
        'merge_last_dim': False,
        'use_validation': True,
        'simulate_confidence_score': True,
        'pretrained_dataset_name': 'h36m',
        'model_prefix': 'MotionAGFormer_',
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
        'n_layers': 16,
        'dim_in': 3,
        'dim_feat': 128,
        'dim_rep': 512,
        'dim_out': 3,
        'mlp_ratio': 4,
        'attn_drop': 0.0,
        'drop': 0.0,
        "drop_path": 0.0,
        "use_layer_scale": True,
        "layer_scale_init_value": 0.00001,
        "use_adaptive_fusion": True,
        "num_heads": 8,
        "qkv_bias": False,
        "qkv_scale": None,
        "hierarchical": False,
        "use_temporal_similarity": True,
        "neighbour_num": 2,
        "temporal_connection_len": 1,
        "use_tcn": False,
        "graph_only": False,
        'classifier_dropout': 0.0,
        'merge_joints': True,
        'classifier_hidden_dims': [1024],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/motionagformer/checkpoint.pth.tr"
    }
    learning_params = {
        'wandb_name': 'MotionAGFormer',
        'experiment_name': '',
        'batch_size': 256,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'AdamW',
        'lr_backbone': 0.0001,
        'lr_head': 0.001,
        'weight_decay': 0.01,
        'lambda_l1': 0.0001,
        'scheduler': "StepLR",
        'lr_decay': 0.99,
        'epochs': 20,
        'stopping_tolerance': 10,
        'lr_step_size': 1
    }

    params = {**param, **data_params, **model_params, **learning_params}

    f = open("./configs/motionagformer/" + f_name, "rb")
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys():
            raise ValueError(
                "Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
        params[p] = new_param[p]

    params['labels_path'] = params['data_path']  # Data Path is the path to csv files by default

    params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0]
    return params, new_param
