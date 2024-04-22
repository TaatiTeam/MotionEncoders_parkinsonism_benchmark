import json

from const import path


def generate_config(param, f_name):
    data_params = {
        'data_type': 'PD',  # options: "Kinect", "GastNet", "PCT", "ViTPose"
        'data_dim': 3,
        'in_data_dim': 2,
        'data_centered': True,
        'merge_last_dim': False,
        'use_validation': True,
        'simulate_confidence_score': False,  # Since poseformerv2 doesn't require confidence score, we just ignore last dim.
        'pretrained_dataset_name': 'h36m',
        'model_prefix': 'poseformerv2_',
        # options: mirror_reflection, random_rotation, random_translation
        # 'augmentation': [],
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'mirror_prob': 0.5,
        'noise_prob': 0.5,
        'axis_mask_prob': 0.5,
        'translation_frac': 0.05,
        'data_norm': "rescaling",
        'select_middle': False,
        'exclude_non_rgb_sequences': False
    }
    model_params = {
        'source_seq_len': 81,
        'num_joints': 17,
        'embed_dim_ratio': 32,
        'depth': 4,
        'number_of_kept_frames': 9,
        'number_of_kept_coeffs': 9,
        'classifier_dropout': 0.5,
        'merge_joints': False,
        'classifier_hidden_dims': [2048],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/poseformerv2/9_81_46.0.bin"
    }
    learning_params = {
        'wandb_name': 'poseformerv2',
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

    f = open("./configs/poseformerv2/" + f_name, "rb")
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys():
            raise ValueError(
                "Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
        params[p] = new_param[p]

    params['labels_path'] = params['data_path']  # Data Path is the path to csv files by default

    params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0]
    return params, new_param
