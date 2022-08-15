import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Preprocess XGBC-HEM benchmark
hem = []
hem_label = []
hem_split = []
label_counter = 1
filenames = ['data/hemolythic/HemoPI-1_model_class1.fasta', 'data/hemolythic/HemoPI-1_model_class0.fasta',
          'data/hemolythic/HemoPI-1_validation_class1.fasta', 'data/hemolythic/HemoPI-1_validation_class0.fasta']
labelnames = [1, 0, 1, 0]
splitnames = ['train', 'train', 'test', 'test']
for f in range(len(filenames)):
    dummy = ''
    with open(filenames[f]) as file:
        for line in file:
            if line.startswith('>'):
                if dummy != '':
                    dummy = ' '.join(dummy)
                    hem.append(dummy.strip())
                    hem_label.append(labelnames[f])
                    hem_split.append(splitnames[f])
                    dummy = ''
            else:
                dummy += line.rstrip('\n')
        dummy = ' '.join(dummy)
        hem.append(dummy.strip())
        hem_label.append(labelnames[f])
        hem_split.append(splitnames[f])

hem_df = pd.DataFrame(data=hem, columns=['text'])
hem_df['labels'] = hem_label
hem_df['split'] = hem_split
hem_df = hem_df.sample(frac=1, random_state=42)
hem_df.to_csv('data/hemolythic/hemolythic.csv', index=False)

hem_df_train = hem_df[hem_df['split'] == 'train'][['text', 'labels']]
hem_df_test = hem_df[hem_df['split'] == 'test'][['text', 'labels']]

print(len(hem_df_train), len(hem_df_test), len(hem_df_train[hem_df_train['labels'] == 1]),
      len(hem_df_train[hem_df_train['labels'] == 0]), len(hem_df_test[hem_df_test['labels'] == 1]),
      len(hem_df_test[hem_df_test['labels'] == 0]))

hem_df_train.to_csv('data/hemolythic/hemolythic_train.csv', index=False)
hem_df_test.to_csv('data/hemolythic/hemolythic_test.csv', index=False)


# Preprocess hlppredfuse benchmark
hem = []
hem_label = []
hem_split = []
label_counter = 1
filenames = ['data/hlppredfuse/Layer1-positive.txt', 'data/hlppredfuse/Layer1-negative.txt',
          'data/hlppredfuse/Layer1-Ind-positive.txt', 'data/hlppredfuse/Layer1-Ind-negative.txt']
labelnames = [1, 0, 1, 0]
splitnames = ['train', 'train', 'test', 'test']
for f in range(len(filenames)):
    dummy = ''
    with open(filenames[f]) as file:
        for line in file:
            if line.startswith('>'):
                if dummy != '':
                    dummy = ' '.join(dummy)
                    hem.append(dummy.strip())
                    hem_label.append(labelnames[f])
                    hem_split.append(splitnames[f])
                    dummy = ''
            else:
                dummy += line.rstrip('\n')
        dummy = ' '.join(dummy)
        hem.append(dummy.strip())
        hem_label.append(labelnames[f])
        hem_split.append(splitnames[f])

hem_df = pd.DataFrame(data=hem, columns=['text'])
hem_df['labels'] = hem_label
hem_df['split'] = hem_split
hem_df = hem_df.sample(frac=1, random_state=42)
hem_df.to_csv('data/hlppredfuse/hlppredfuse.csv', index=False)

hem_df_train = hem_df[hem_df['split'] == 'train'][['text', 'labels']]
hem_df_test = hem_df[hem_df['split'] == 'test'][['text', 'labels']]

print(len(hem_df_train), len(hem_df_test), len(hem_df_train[hem_df_train['labels'] == 1]),
      len(hem_df_train[hem_df_train['labels'] == 0]), len(hem_df_test[hem_df_test['labels'] == 1]),
      len(hem_df_test[hem_df_test['labels'] == 0]))

hem_df_train.to_csv('data/hlppredfuse/hlppredfuse_train.csv', index=False)
hem_df_test.to_csv('data/hlppredfuse/hlppredfuse_test.csv', index=False)



# Preprocess RNN-hem benchmark
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromFASTA, MolToSmiles
from rdkit.Chem import Descriptors


def seq_to_smiles(seq):
    mol = MolFromFASTA(seq, flavor=True, sanitize = True)
    smiles = MolToSmiles(mol, isomericSmiles=True)
    return smiles


def MW(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(mol)
    return mw


df_raw = pd.read_csv('data/rnnamp/peptides_complete.csv')

print(df_raw.columns)
hem_columns = []
for c in df_raw.columns:
    if 'hem' in c.lower():
        hem_columns.append(c)


df = df_raw.dropna(subset=['HEMOLITIC CYTOTOXIC ACTIVITY - UNIT'])
print('HEMOLITIC CYTOTOXIC ACTIVITY - TARGET CELL', df_raw['HEMOLITIC CYTOTOXIC ACTIVITY - TARGET CELL'].unique())
df = df[df['HEMOLITIC CYTOTOXIC ACTIVITY - TARGET CELL'] == 'Human erythrocytes']
df.dropna(subset=['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'], inplace=True)
conc = []
for i in range(len(df)):
    dummy_unit = df.iloc[i]['HEMOLITIC CYTOTOXIC ACTIVITY - UNIT']

    dummy = df.iloc[i]['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION']
    dummy = dummy.replace('>', '')
    dummy = dummy.replace('<', '')
    dummy = dummy.replace('=', '')
    dummy = dummy.replace('up to ', '')
    # dummy = dummy.replace('>', '')
    corrected_dummy = 0
    if '±' in dummy:
        corrected_dummy = dummy.split('±')[0]
    elif '-' in dummy:
        corrected_dummy = dummy.split('-')[0]
    elif '(' in dummy:
        corrected_dummy = dummy.split('(')[0]

    else:
        corrected_dummy = dummy
    if dummy_unit == "µg/ml":
        try:
            mw = MW(seq_to_smiles(df.iloc[i]['SEQUENCE']))
        except:
            conc.append(None)
            continue
        corrected_dummy = float(corrected_dummy) / (mw / 1000)
    conc.append(float(corrected_dummy))
df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = conc
df.dropna(subset=['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'], inplace=True)
df = df.astype({'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION': float})
df_50 = df[df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] >= 50]
df_non_hem = df_50[df_50['HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS GROUP'].isin(['0-10% Hemolysis', '10-20% Hemolysis'])]
df_hem = df[df['HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS GROUP'].isin(['20-30% Hemolysis', '30-40% Hemolysis', '40-50% Hemolysis', '50-60% Hemolysis', '60-70% Hemolysis', '70-80% Hemolysis', '80-90% Hemolysis', '90-100% Hemolysis'])]

print(len(df_raw), len(df), len(df_50), len(df_non_hem), len(df_hem))

seqs_hem = list(np.array(df_hem['SEQUENCE']))
seqs_hem = [s.upper() for s in seqs_hem]
seqs_non_hem = list(np.array(df_non_hem['SEQUENCE']))
seqs_non_hem = [s.upper() for s in seqs_non_hem]

df_ref = pd.read_csv('data/rnnamp/DAASP_RNN_dataset.csv')
seqs_ref = list(np.array(df_ref['Sequence']))
seqs_ref = [s.upper() for s in seqs_ref]

hem_counter = 0
non_hem_counter = 0
for s in seqs_ref:
    if s in seqs_non_hem:
        non_hem_counter += 1
    elif s in seqs_hem:
        hem_counter += 1


print(hem_counter, non_hem_counter)

joint_seqs = []
original_seqs = []
labels = []
splits = []
for i in range(len(df_ref)):
    current_n = df_ref.iloc[i]['N terminus']
    current_seq = df_ref.iloc[i]['Sequence']
    current_seq = current_seq.upper()
    current_c = df_ref.iloc[i]['C terminus']
    current_split = df_ref.iloc[i]['Set']
    if current_split == 'training':
        current_split = 'train'
    if pd.isna(current_n):
        current_n = ''
    if pd.isna(current_c):
        current_c = ''
    joint_seq = current_n + current_seq + current_c
    joint_seq = joint_seq.upper()
    if current_seq in seqs_non_hem:
        joint_seq = ' '.join(joint_seq)
        joint_seqs.append(joint_seq.strip())
        current_seq = ' '.join(current_seq)
        original_seqs.append(current_seq.strip())
        labels.append(0)
        splits.append(current_split)
    elif current_seq in seqs_hem:
        joint_seq = ' '.join(joint_seq)
        joint_seqs.append(joint_seq.strip())
        current_seq = ' '.join(current_seq)
        original_seqs.append(current_seq.strip())
        labels.append(1)
        splits.append(current_split)

hem_df = pd.DataFrame(data=list(zip(joint_seqs, original_seqs, labels, splits)), columns=['text', 'original_sequence', 'labels', 'split'])
# hem_df = pd.DataFrame(data=list(zip(original_seqs, labels, splits)), columns=['text', 'labels', 'split'])
hem_df = hem_df.sample(frac=1, random_state=42)
hem_df.to_csv('data/rnnamp/rnnamp.csv', index=False)

hem_df_train = hem_df[hem_df['split'] == 'train'][['text', 'labels']]
hem_df_test = hem_df[hem_df['split'] == 'test'][['text', 'labels']]

print(len(hem_df_train), len(hem_df_test), len(hem_df_train[hem_df_train['labels'] == 1]),
      len(hem_df_train[hem_df_train['labels'] == 0]), len(hem_df_test[hem_df_test['labels'] == 1]),
      len(hem_df_test[hem_df_test['labels'] == 0]))

hem_df_train.to_csv('data/rnnamp/rnnamp_train.csv', index=False)
hem_df_test.to_csv('data/rnnamp/rnnamp_test.csv', index=False)


# find overlap between datasets
df_rnn = pd.read_csv('data/rnnamp/rnnamp.csv')
df_hlp = pd.read_csv('data/hlppredfuse/hlppredfuse.csv')
df_hem = pd.read_csv('data/hemolythic/hemolythic.csv')


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


for df1 in [df_rnn, df_hlp, df_hem]:
    for df2 in [df_rnn, df_hlp, df_hem]:
        seq1 = list(df1['text'])
        seq2 = list(df2['text'])
        intersect = intersection(seq1, seq2)
        print(len(intersect), len(seq1), len(seq2))


# save as fasta for cd-hit analysis
df_train = pd.read_csv('data/hlppredfuse/hlppredfuse_train.csv')
df_test = pd.read_csv('data/hlppredfuse/hlppredfuse_test.csv')

train_seqs = [s.replace(" ", "") for s in list(df_train['text'])]
test_seqs = [s.replace(" ", "") for s in list(df_test['text'])]

train_fasta = []
test_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')
for i in range(len(test_seqs)):
    test_fasta.append('>seq'+str(i)+'\n')
    test_fasta.append(test_seqs[i]+'\n')

with open('data/hlppredfuse/hlppredfuse_train.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)
with open('data/hlppredfuse/hlppredfuse_test.fasta', 'w+') as f:
    for line in test_fasta:
        f.write(line)
# cd-hit-2d -i ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_train.fasta -i2 ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_test.fasta -o ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_cdhit_results -c 0.4 -n 2
# cd-hit-2d -i ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_test.fasta -i2 ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_train.fasta -o ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_cdhit_results -c 0.4 -n 2

test_cdhit = []
with open('data/hlppredfuse/hlppredfuse_cdhit_results', 'r') as f:
    for line in f:
        test_cdhit.append(line)

remaining_idx = []
for i in test_cdhit:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_test_df = df_test.iloc[remaining_idx]
print(len(df_test), len(remaining_test_df))

remaining_train_df = df_train.iloc[remaining_idx]
print(len(df_train), len(remaining_train_df))


df = pd.read_csv('data/hlppredfuse/hlppredfuse.csv')
train_seqs = [s.replace(" ", "") for s in list(df['text'])]
train_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')

with open('data/hlppredfuse/hlppredfuse.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)

# cd-hit -i ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse.fasta -o ../../PycharmProjects/ampdeep/data/hlppredfuse/hlppredfuse_cdhit_results -c 0.4 -n 2

cdhit_results = []
with open('data/hlppredfuse/hlppredfuse_cdhit_results', 'r') as f:
    for line in f:
        cdhit_results.append(line)

remaining_idx = []
for i in cdhit_results:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_df = df.iloc[remaining_idx]
print(len(df), len(remaining_df))

# hlpprefuse 3518 1141
# rnnhem 2557 140
# XGBC 1104 357


df = pd.read_csv('data/rnnamp/rnnamp.csv')
train_seqs = [s.replace(" ", "") for s in list(df['text'])]
train_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')

with open('data/rnnamp/rnnamp.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)

# cd-hit -i ../../PycharmProjects/ampdeep/data/rnnamp/rnnamp.fasta -o ../../PycharmProjects/ampdeep/data/rnnamp/rnnamp_cdhit_results -c 0.4 -n 2

cdhit_results = []
with open('data/rnnamp/rnnamp_cdhit_results', 'r') as f:
    for line in f:
        cdhit_results.append(line)

remaining_idx = []
for i in cdhit_results:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_df = df.iloc[remaining_idx]
print(len(df), len(remaining_df))



df = pd.read_csv('data/hemolythic/hemolythic.csv')
train_seqs = [s.replace(" ", "") for s in list(df['text'])]
train_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')

with open('data/hemolythic/hemolythic.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)

# cd-hit -i ../../PycharmProjects/ampdeep/data/hemolythic/hemolythic.fasta -o ../../PycharmProjects/ampdeep/data/hemolythic/hemolythic_cdhit_results -c 0.4 -n 2

cdhit_results = []
with open('data/hemolythic/hemolythic_cdhit_results', 'r') as f:
    for line in f:
        cdhit_results.append(line)

remaining_idx = []
for i in cdhit_results:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_df = df.iloc[remaining_idx]
print(len(df), len(remaining_df))

# combine all hemolythic datasets
df_rnn = pd.read_csv('data/rnnamp/rnnamp.csv')
df_hlp = pd.read_csv('data/hlppredfuse/hlppredfuse.csv')
df_hem = pd.read_csv('data/hemolythic/hemolythic.csv')

all_dfs = [df_rnn, df_hlp, df_hem]

seqs = []
labels = []
for df in all_dfs:
    seqs.extend(list(np.array(df['text'])))
    labels.extend(list(np.array(df['labels'])))

merged = pd.DataFrame(data=zip(seqs, labels), columns=['text', 'labels'])
merged = merged.sample(frac=1, random_state=42)
merged.to_csv('data/combined/combined.csv', header=True, index=False)

train_seqs = [s.replace(" ", "") for s in list(merged['text'])]
train_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')

with open('data/combined/combined.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)

# cd-hit -i ../../PycharmProjects/ampdeep/data/combined/combined.fasta -o ../../PycharmProjects/ampdeep/data/combined/combined_cdhit_results -c 0.4 -n 2

cdhit_results = []
with open('data/combined/combined_cdhit_results', 'r') as f:
    for line in f:
        cdhit_results.append(line)

remaining_idx = []
for i in cdhit_results:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_merged = merged.iloc[remaining_idx]
print(len(merged), len(remaining_merged))
print(np.sum(remaining_merged['labels']))


ind_test_pos = remaining_merged[remaining_merged['labels'] == 1].sample(n=50, random_state=42)
ind_test_neg = remaining_merged[remaining_merged['labels'] == 0].sample(n=50, random_state=42)

ind_test = pd.concat([ind_test_pos, ind_test_neg])
ind_test = ind_test.sample(frac=1, random_state=42)
ind_test.reset_index(drop=True, inplace=True)

ind_test.to_csv('data/combined/combined_test.csv', header=True, index=False)

# create fasta file from indepent test set
train_seqs = [s.replace(" ", "") for s in list(ind_test['text'])]
train_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')

with open('data/combined/combined_test.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)

# cd-hit-2d -i ../../PycharmProjects/ampdeep/data/combined/combined_test.fasta -i2 ../../PycharmProjects/ampdeep/data/combined/combined.fasta -o ../../PycharmProjects/ampdeep/data/combined/combined_cdhit_results -c 0.4 -n 2

test_cdhit = []
with open('data/combined/combined_cdhit_results', 'r') as f:
    for line in f:
        test_cdhit.append(line)

remaining_idx = []
for i in test_cdhit:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_merged = merged.iloc[remaining_idx]
print(len(merged), len(remaining_merged))
print(np.sum(remaining_merged[remaining_merged['labels'] == 1]))


ind_train_pos = remaining_merged[remaining_merged['labels'] == 1]
ind_train_neg = remaining_merged[remaining_merged['labels'] == 0].sample(n=len(ind_train_pos), random_state=42)

ind_train = pd.concat([ind_train_pos, ind_train_neg])
ind_train = ind_train.sample(frac=1, random_state=42)
ind_train.reset_index(drop=True, inplace=True)

ind_train.to_csv('data/combined/combined_train.csv', header=True, index=False)

ind_train = pd.read_csv('data/combined/combined_train.csv')
ind_test = pd.read_csv('data/combined/combined_test.csv')
print(len(ind_train), len(ind_train[ind_train['labels'] == 1]), len(ind_train[ind_train['labels'] == 0]))
print(len(ind_test), len(ind_test[ind_test['labels'] == 1]), len(ind_test[ind_test['labels'] == 0]))


# hlpprefuse 3518 1141
# rnnhem 2557 140
# XGBC 1104 357



# plot a Stacked Bar Chart using matplotlib
import matplotlib.pyplot as plt
similarity_df = pd.DataFrame(data=[['RNN-Hem', 140, 2557-140],
                                   ['HLPpred-Fuse', 1141, 3518-1141],
                                   ['XGBC-Hem', 357, 1104-357]],
                             columns=['Dataset', 'representative', 'redundant'])
similarity_df['total'] = similarity_df["representative"] + similarity_df["redundant"]
similarity_df['Redundant'] = similarity_df['redundant']/similarity_df['total']*100
similarity_df['Non-Redundant'] = similarity_df['representative']/similarity_df['total']*100


similarity_df_total = similarity_df['total']
similarity_df.drop(['total', 'redundant', 'representative'], axis=1, inplace=True)
similarity_df_rel = similarity_df[similarity_df.columns[1:]]

fontsize = 15
# fig = plt.figure(dpi=300)
similarity_df.plot(
    x='Dataset',
    kind='barh',
    stacked=True,
    # title='Percentage Stacked Bar Graph',
    mark_right=True)
for n in similarity_df_rel:
    for i, (cs, ab, pc) in enumerate(zip(similarity_df.iloc[:, 1:].cumsum(1)[n],
                                         similarity_df[n], similarity_df_rel[n])):
        print(cs, ab, pc)
        plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%',
                 va='center', ha='center', fontsize=fontsize)
plt.ylabel('Dataset', fontsize=fontsize+2)
plt.xlabel('Percentage', fontsize=fontsize+2)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim(-0.5, 2.7)
plt.legend(fontsize=fontsize-2, ncol=2)
plt.tight_layout()
plt.savefig('results/similarity.png', format='png', dpi=300)
plt.show()







