import os
import sys
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_stats(video_names, predictions, rep_out, type):
    with open(os.path.join(rep_out, f'stats_output_{type}.txt'), 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f 
        data = pd.DataFrame({'VideoName': video_names, 'predScore': predictions})

        data['Participant'] = data['VideoName'].apply(lambda x: x.split('_')[0])
        data['MedicationStatus'] = data['VideoName'].apply(lambda x: x.split('_')[1])

        on_medication_data = data[data['MedicationStatus'] == 'on']
        mean_scores_on = on_medication_data.groupby('Participant')['predScore'].mean()
        
        off_medication_data = data[data['MedicationStatus'] == 'off']
        mean_scores_off = off_medication_data.groupby('Participant')['predScore'].mean()

        print(mean_scores_on) 
        print(mean_scores_off)
        
        participants_on = set(mean_scores_on.index)
        participants_off = set(mean_scores_off.index)
        common_participants = participants_on.intersection(participants_off)
        mean_scores_on_paired = mean_scores_on[mean_scores_on.index.isin(common_participants)]
        mean_scores_off_paired = mean_scores_off[mean_scores_off.index.isin(common_participants)]
            
        stat, p = stats.shapiro(mean_scores_on_paired)
        print('Shapiro-Wilk Test statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('ON Sample looks Gaussian (fail to reject H0)')
        else:
            print('ON Sample does not look Gaussian (reject H0)')

        stat, p = stats.shapiro(mean_scores_off_paired)
        print('Shapiro-Wilk Test statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('OFF Sample looks Gaussian (fail to reject H0)')
        else:
            print('OFF Sample does not look Gaussian (reject H0)')

        if mean_scores_on_paired.index.equals(mean_scores_off_paired.index):
            print("Datasets are properly paired.")
        else:
            print("Datasets are not properly paired.")
    
        stat, p_value = stats.wilcoxon(mean_scores_on_paired, mean_scores_off_paired)
        print(f'Wilcoxon Signed-Rank Test Statistic: {stat}')
        print(f'P-Value: {p_value}')
        
        t_statistic, p_value = stats.ttest_rel(mean_scores_on_paired, mean_scores_off_paired)
        print(f"t-test: T-Statistic: {t_statistic}")
        print(f"P-Value: {p_value}")
        
        sys.stdout = original_stdout
        
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Histogram for ON medication on the left subplot
    sns.histplot(mean_scores_on_paired, kde=True, bins=[-0.3, 0,0.3,0.7,1,1.3, 1.7, 2,2.3], color='blue', label='ON Medication', ax=axes[0])
    axes[0].set_title('ON Medication')
    axes[0].set_xlabel('UPDRS Scores')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Histogram for OFF medication on the right subplot
    sns.histplot(mean_scores_off_paired, kde=True, bins=[-0.3, 0,0.3,0.7,1,1.3, 1.7, 2,2.3], color='red', label='OFF Medication', ax=axes[1])
    axes[1].set_title('OFF Medication')
    axes[1].set_xlabel('UPDRS Scores')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(os.path.join(rep_out, f'pred_distribution_histogram_{type}.png'))
    plt.close()