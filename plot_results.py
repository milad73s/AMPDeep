import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parameter_dict = {}
parameter_dict['full_finetune'] = {
    'freeze_positional': 0,
    'freeze_non_positional': 0,
    'freeze_attention': 0,
    'freeze_layer_norm': 0,
    'freeze_pooler': 0,
    'use_pooler': 1,
    'use_mean': 0
}
parameter_dict['feature_extractor'] = {
    'freeze_positional': 1,
    'freeze_non_positional': 1,
    'freeze_attention': 1,
    'freeze_layer_norm': 1,
    'freeze_pooler': 1,
    'use_pooler': 1,
    'use_mean': 0
}
parameter_dict['pooler_finetune'] = {
    'freeze_positional': 1,
    'freeze_non_positional': 1,
    'freeze_attention': 1,
    'freeze_layer_norm': 1,
    'freeze_pooler': 0,
    'use_pooler': 1,
    'use_mean': 0
}
parameter_dict['pooler_replacement'] = {
    'freeze_positional': 1,
    'freeze_non_positional': 1,
    'freeze_attention': 1,
    'freeze_layer_norm': 1,
    'use_pooler': 0
}
parameter_dict['input_embedding'] = {
    'freeze_positional': 0,
    'freeze_non_positional': 0,
    'freeze_attention': 1,
    'freeze_layer_norm': 1,
}
parameter_dict['positional_embedding'] = {
    'freeze_positional': 0,
    'freeze_non_positional': 1,
    'freeze_attention': 1,
    'freeze_layer_norm': 1,
}
parameter_dict['positional_embedding_layer_norm'] = {
    'freeze_positional': 0,
    'freeze_non_positional': 1,
    'freeze_attention': 1,
    'freeze_layer_norm': 0,
}
parameter_dict['input_embedding_layer_norm'] = {
    'freeze_positional': 0,
    'freeze_non_positional': 0,
    'freeze_attention': 1,
    'freeze_layer_norm': 0,
}

all_run_type = ['full_finetune', 'feature_extractor', 'pooler_finetune',
                'pooler_replacement', 'input_embedding', 'positional_embedding', 'input_embedding_layer_norm',
                'positional_embedding_layer_norm']
all_subject = ['hemolythic', 'hlppredfuse', 'rnnamp', 'combined']

results = {}
for subject in all_subject:
    results[subject] = {}


missing = 0
found = 0
for subject in all_subject:
    for run_type in all_run_type:
        results_df = pd.read_csv('results/training_results.csv')
        results_df = results_df[results_df['subject'] == subject]

        current_params = parameter_dict[run_type]
        results_df = results_df[results_df['frozenpositional'] == current_params['freeze_positional']]
        results_df = results_df[results_df['frozennonpositional'] == current_params['freeze_non_positional']]
        results_df = results_df[results_df['frozenattention'] == current_params['freeze_attention']]
        results_df = results_df[results_df['frozenlayernorm'] == current_params['freeze_layer_norm']]
        if 'freeze_pooler' in current_params:
            results_df = results_df[results_df['frozenpooler'] == current_params['freeze_pooler']]
        if 'use_pooler' in current_params:
            results_df = results_df[results_df['usepooler'] == current_params['use_pooler']]
        if 'use_mean' in current_params:
            results_df = results_df[results_df['usemean'] == current_params['use_mean']]

        if len(results_df) == 0:
            missing += 1
            results[subject][run_type] = 0.3 # placeholder
            continue
        found += 1

        results_df.reset_index(drop=True, inplace=True)
        print(subject, run_type, len(results_df),results_df.iloc[results_df['mcc_eval'].idxmax()]['mcc_test'])
        results[subject][run_type] = results_df.iloc[results_df['mcc_eval'].idxmax()]['mcc_test']
print(missing, found)

subject_map = {'hemolythic': 'XGBC-Hem', 'hlppredfuse':'HLPpred-Fuse', 'rnnamp': 'RNN-Hem', 'combined': 'Combined'}
marker_map = {'hemolythic': 'v', 'hlppredfuse': '+', 'rnnamp': '^', 'combined': '*'}
fontsize = 15
fig = plt.figure(dpi=300)
average_results = []
for subject in all_subject:
    current_results = [results[subject][run_type] for run_type in all_run_type]
    plt.plot(np.arange(1, len(current_results)+1), current_results, linestyle='dashed',
             marker=marker_map[subject], label=subject_map[subject], alpha=0.5)
    average_results.append(current_results)
average_results = np.array(average_results)
plt.plot(np.arange(1,len(average_results[0]) + 1), np.mean(average_results, axis=0), 'o-', label='Average performance', c='black')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim([-0.19,1.1])
# plt.xticks(np.arange(1,len(average_results[0]) + 1), ['Scenario ' + str(i) for i in np.arange(1,len(average_results[0]) + 1)])
plt.xlabel('Fine-tuning scenario number', fontsize=fontsize)
plt.ylabel('Test MCC', fontsize=fontsize+2)
plt.legend(title='Dataset:', fontsize=fontsize-6, title_fontsize=fontsize-6)
plt.tight_layout()
plt.savefig('results/selective_finetuning_results.png', format='png', dpi=300)
plt.show()
