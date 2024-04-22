import joblib
import pkg_resources
import wandb
import datetime

from sklearn.metrics import classification_report, confusion_matrix

from data.dataloaders import *
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from training_hypertune import train_model, final_test
from utility import utils
from const import path
from eval_encoder_hypertune import log_results
from stat_analysis.get_stats import *

from torch import nn

def map_to_classifier_dim(backbone_name, op):
    #SET approperiate classifier dim according to the baseline feature dim
    if backbone_name == 'poseformer':
        classifier_hidden_dims_choices = {   #POTR
        'option1': [],
        }
    elif backbone_name == 'motionbert':
        classifier_hidden_dims_choices = {   #motionBert
            'option1': [],
        }
    elif backbone_name == 'poseformerv2':
        classifier_hidden_dims_choices = {'option1': []}
    elif backbone_name == 'mixste':
        classifier_hidden_dims_choices = {'option1': []}
    elif backbone_name == 'motionagformer':
        classifier_hidden_dims_choices = {'option1': []}
    return classifier_hidden_dims_choices[op]

def test__hypertune(params, new_params, all_folds, backbone_name, device):   
    try:
        exp_path = path.OUT_PATH + os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
        study_path =  path.OUT_PATH + os.path.join(params['model_prefix'], str(params['readstudyfrom']))
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            last_fold_model = sorted(os.listdir(os.path.join(exp_path, 'models')), key=utils.natural_sort_key)
            if len(last_fold_model) > 0:
                last_fold_model = last_fold_model[-1]
                all_folds = range(int(last_fold_model.split('fold')[-1]), all_folds[-1] + 1)
            
        params['model_prefix'] = params['model_prefix'] + '/' + str(params['last_run_foldnum'])

        print('Reading study from '+ study_path + '_study/study_mid.pkl' )
        study = joblib.load(study_path + '_study/study_mid.pkl')

        best_params = study.best_trial.params
        best_params['best_trial_number'] = study.best_trial.number  

        # Get the top 10 trials
        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:30] #change to get the best X trials
        for i, trial in enumerate(top_trials):
            print(f"best{i}: Trial {trial.number}, Value: {trial.value}, Params: {trial.params}")
    except:
        print("**STUDY NOT FOUND: Running test with some default params**")
        variations = []
        changes = ['']
        best_params = {
                "lr": 1e-05,
                "num_epochs": 20,
                "num_hidden_layers": 2,
                "layer_sizes":[256, 50, 16, 3],
                # "architecture_type": architecture_type,
                "optimizer": 'RMSprop',
                "use_weighted_loss": True,
                "batch_size": 128,
                # "use_batch_norm": False,
                "dropout_rate": 0.1,
                'weight_decay': 0.00057,
                'momentum': 0.66
                }
        variations.append(best_params)
        
    print("====================================BEST MODEL====================================================")
    print("========================================================================================")
    print(f"Trial {best_params['best_trial_number']}, lr: {best_params['lr']}, num_epochs: {best_params['num_epochs']}")
    print(f"classifier_hidden_dims: {map_to_classifier_dim(backbone_name, best_params['classifier_hidden_dims'])}")
    print(f"optimizer_name: {best_params['optimizer']}, use_weighted_loss: {best_params['use_weighted_loss']}")
    print(f"batch_size: {params['batch_size']}, classifier_dropout: {best_params['dropout_rate']}")
    print("========================================================================================")
    print("========================================================================================")
        
    params['classifier_dropout'] = best_params['dropout_rate']
    params['classifier_hidden_dims'] = map_to_classifier_dim(backbone_name, best_params['classifier_hidden_dims'])
    params['optimizer'] = best_params['optimizer']
    params['lr_head'] = best_params['lr']
    # params['lr_decay'] = best_params['lr_decay']
    # params['lr_step_size'] = best_params['lr_step_size']
    params['lambda_l1'] = best_params['lambda_l1']
    params['epochs'] = best_params['num_epochs']
    if best_params['use_weighted_loss']:
        params['criterion'] = 'WCELoss'
    else:
        params['criterion'] = 'CrossEntropyLoss'    
    if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop']:
        params['weight_decay'] =  best_params['weight_decay']
    if params['optimizer'] == 'SGD':
        params['momentum'] = best_params['momentum']
    
    params['wandb_name'] = params['wandb_name'] + '_test' + str(params['last_run_foldnum'])
    wandb.init(name=params['wandb_name'],
            project='MotionEncoderEvaluator_PD',
            settings=wandb.Settings(start_method='fork'))
    wandb.config.update(params)
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update({'installed_packages': installed_packages})
    wandb.config.update({'new_params': new_params})
    
    rep_out = path.OUT_PATH + os.path.join(params['model_prefix'])
    
    splits = []
    for fold in all_folds:
        train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name, fold)
        splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))
    
    #if start training from middle
    if all_folds[0] > 1:
        with open(os.path.join(rep_out, f'total_results_fold{all_folds[0]-1}.pkl'), 'rb') as file:
            total_results = pickle.load(file)
        total_video_names = total_results['total_video_names'].tolist()
        total_outs_best = total_results['total_outs_best'].tolist()
        total_outs_last = total_results['total_outs_last'].tolist()
        total_states = total_results['total_states'].tolist()
        total_gts = total_results['total_gts'].tolist()
        total_logits = []
    else:
        total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits, start=all_folds[0]):
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
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        if fold == 1:
            model_params = count_parameters(model)
            print(f"[INFO] Model has {model_params} parameters.")
        
        train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name)
        # Evaluate the model
        # val_loss, val_acc, val_f1_score = validate_model(model, val_dataset_fn, params, class_weights)
        checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'],'models', f"fold{fold}")
        best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
        load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)['model'])
        model.cuda()
        outs, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
        total_outs_best.extend(outs)
        total_gts.extend(gts)
        total_states.extend(states)
        total_video_names.extend(video_names)
        print(f'fold # of test samples: {len(video_names)}')
        print(f'current sum # of test samples: {len(total_video_names)}')
        attributes = [total_outs_best, total_gts]
        names = ['predicted_classes', 'true_labels']
        res_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'results')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        utils.save_json(os.path.join(res_dir, 'results_Best_fold{}.json'.format(fold)), attributes, names)
        

        total_logits.extend(logits)
        attributes = [total_logits, total_gts]

        logits_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'logits')
        if not os.path.exists(logits_dir):
            os.makedirs(logits_dir)
        utils.save_json(os.path.join(logits_dir, 'logits_Best_fold{}.json'.format(fold)), attributes, names)
        

        last_ckpt_path = os.path.join(checkpoint_root_path, 'latest_epoch.pth.tr')
        load_pretrained_weights(model, checkpoint=torch.load(last_ckpt_path)['model'])
        model.cuda()
        outs_last, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
        total_outs_last.extend(outs_last)
        attributes = [total_outs_last, total_gts]
        utils.save_json(os.path.join(res_dir, 'results_last_fold{}.json'.format(fold)), attributes, names)
        
        res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last, 'total_gts':total_gts, 'total_states':total_states})
        with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
            pickle.dump(res, file)

        
        end_time = datetime.datetime.now()
        
        duration = end_time - start_time
        print(f"Fold {fold} run time:", duration)

    
    #================================BEST REPORTS=============================
    rep_best_final = classification_report(total_gts, total_outs_best)  
    confusion_best_final = confusion_matrix(total_gts, total_outs_best)
    log_results(rep_best_final, confusion_best_final, 'best_report_allfolds.txt', 'best_confusion_matrix_allfolds.png', rep_out)
    
    #==============ON - OFF REPORTS===================
    # Splitting the data into 'ON' and 'OFF' groups
    total_gts_on = [gt for gt, state in zip(total_gts, total_states) if state == 'ON']
    total_outs_best_on = [out for out, state in zip(total_outs_best, total_states) if state == 'ON']

    total_gts_off = [gt for gt, state in zip(total_gts, total_states) if state == 'OFF']
    total_outs_best_off = [out for out, state in zip(total_outs_best, total_states) if state == 'OFF']

    # Calculating metrics for the 'ON' group
    print("============ON==================")
    rep_best_final_on = classification_report(total_gts_on, total_outs_best_on)
    confusion_best_final_on = confusion_matrix(total_gts_on, total_outs_best_on)
    log_results(rep_best_final_on, confusion_best_final_on, 'best_report_allfolds_ON.txt', 'best_confusion_matrix_allfolds_ON.png', rep_out)

    print("============OFF==================")
    # Calculating metrics for the 'OFF' group
    rep_best_final_off = classification_report(total_gts_off, total_outs_best_off)
    confusion_best_final_off = confusion_matrix(total_gts_off, total_outs_best_off)
    log_results(rep_best_final_off, confusion_best_final_off, 'best_report_allfolds_OFF.txt', 'best_confusion_matrix_allfolds_OFF.png', rep_out)
    
    
    #================================LAST REPORTS=============================
    rep_last_final = classification_report(total_gts, total_outs_last)  
    confusion_last_final = confusion_matrix(total_gts, total_outs_last)
    log_results(rep_last_final, confusion_last_final, 'last_report_allfolds.txt', 'last_confusion_matrix_allfolds.png', rep_out)
    
    #==============ON - OFF REPORTS===================
    # Splitting the data into 'ON' and 'OFF' groups
    total_gts_on = [gt for gt, state in zip(total_gts, total_states) if state == 'ON']
    total_outs_last_on = [out for out, state in zip(total_outs_last, total_states) if state == 'ON']

    total_gts_off = [gt for gt, state in zip(total_gts, total_states) if state == 'OFF']
    total_outs_last_off = [out for out, state in zip(total_outs_last, total_states) if state == 'OFF']

    # Calculating metrics for the 'ON' group
    print("============ON==================")
    rep_last_final_on = classification_report(total_gts_on, total_outs_last_on)
    confusion_last_final_on = confusion_matrix(total_gts_on, total_outs_last_on)
    log_results(rep_last_final_on, confusion_last_final_on, 'last_report_allfolds_ON.txt', 'last_confusion_matrix_allfolds_ON.png', rep_out)

    print("============OFF==================")
    # Calculating metrics for the 'OFF' group
    rep_last_final_off = classification_report(total_gts_off, total_outs_last_off)
    confusion_last_final_off = confusion_matrix(total_gts_off, total_outs_last_off)
    log_results(rep_last_final_off, confusion_last_final_off, 'last_report_allfolds_OFF.txt', 'last_confusion_matrix_allfolds_OFF.png', rep_out)
    
    res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last, 'total_gts':total_gts})
    with open(os.path.join(rep_out, 'final_results.pkl'), 'wb') as file:
        pickle.dump(res, file)
        
    with open(path.OUT_PATH + os.path.join(params['model_prefix'], 'final_results.pkl'), 'rb') as file:
        final_results = pickle.load(file)
    total_video_names = final_results['total_video_names']
    total_outs_best = final_results['total_outs_best']
    total_outs_last = final_results['total_outs_last']
    rep_out = path.OUT_PATH + os.path.join(params['model_prefix'])
    
    get_stats(total_video_names, total_outs_best, rep_out, 'best')
    get_stats(total_video_names, total_outs_last, rep_out, 'last')
    wandb.finish()