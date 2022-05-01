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
