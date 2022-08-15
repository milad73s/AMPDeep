import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file = 'data/swissprot/swissprot.txt'

keywords = []
seqs = []
ids = []
flag = 0
sq_counter = 0
with open(file) as f:
    counter = 0
    dummy_kw = []
    dummy_sq = []
    dummy_id = []
    previous_starter = ''
    for line in f:
        # Extract the IDs
        if line.startswith('AC'):
            dummy_id = line.lstrip('AC').strip().split(';')
        # Extract the keywordsd
        if line.startswith('KW'):
            dummy = line.rstrip('\n').lstrip('KW').rstrip(';').split(';')
            dummy_cleaned = []
            for k in dummy:
                dummy_cleaned.append(k.strip(' ').rstrip('.'))
            dummy_kw.extend(dummy_cleaned)
        # Extract the sequence
        if line.startswith('SQ'):
            found_sequence = 1
            dummy_cleaned_sq = ''
        # continue with the sequence in the next lines
        if line.startswith('  ') and found_sequence == 1:
            dummy_sq = line.rstrip('\n').lstrip('SQ').rstrip().lstrip().split()
            dummy_cleaned_sq += ''.join(d for d in dummy_sq)
        # Save the sequence
        if line.startswith('//') and found_sequence == 1:
            sq_counter += 1
            ids.append(dummy_id)
            dummy_id = []
            keywords.append(dummy_kw)
            dummy_kw = []
            seqs.append(dummy_cleaned_sq)
            found_sequence = 0
        previous_starter = line[:2]
        if counter % 1000000 == 0:
            print(counter)
        counter += 1


df = pd.DataFrame(seqs, columns=['sequence'])
df['length'] = [len(s) for s in df['sequence']]
keywords_str = []
for k in keywords:
    dummy = ''
    for k_ in k:
        dummy += k_ + ','
    keywords_str.append(dummy.rstrip(','))
df['keyword'] = keywords_str

ids_str = []
for k in ids:
    dummy = ''
    for k_ in k:
        dummy += k_ + ','
    ids_str.append(dummy.rstrip(','))
df['id'] = ids_str
df.drop_duplicates(subset=['sequence'], inplace=True, keep='first')

# Extract the peptides
df_pep = df[df['length'] <= 200]

# preprocess Secreted
secreted_idx = []
for i in range(len(df_pep)):
    current_keywords = df_pep.iloc[i]['keyword'].split(',')
    dummy = False
    for k in current_keywords:
        if k == 'Secreted':
            dummy = True
    secreted_idx.append(dummy)
df_secreted = df_pep[secreted_idx]
df_secreted['label'] = 1

# Find non-secretory peptides from peptides within the cytoplasm with no secretory label
df_non_secreted = df_pep[list(~np.array(secreted_idx))]
df_non_secreted['label'] = 0
cytoplasm_idx = []
for i in range(len(df_non_secreted)):
    current_keywords = df_non_secreted.iloc[i]['keyword'].split(',')
    dummy = False
    for k in current_keywords:
        if k == 'Cytoplasm':
            dummy = True
    cytoplasm_idx.append(dummy)
df_cytoplasm = df_non_secreted[cytoplasm_idx]
df_cytoplasm_non_secretory = df_cytoplasm.sample(n=len(df_secreted), random_state=42)
print(len(df), len(df_pep), len(df_secreted), len(df_non_secreted), len(df_cytoplasm))

df_secreted_all = pd.concat([df_secreted, df_cytoplasm_non_secretory])
print(np.sum(df_secreted_all['label']), len(df_secreted_all))
spaced_seqs = []
counter = 0
for line in df_secreted_all['sequence']:
    d = ' '.join(line)
    spaced_seqs.append(d.strip())
    if counter % 10000 == 0:
        print(counter)
    counter += 1
df_secreted_all['sequence'] = spaced_seqs
df_secreted_all = df_secreted_all.sample(frac=1, random_state=42)
df_secreted_all.rename(columns={'sequence': 'text', 'label':'labels'}, inplace=True)
df_secreted_all.to_csv('data/secreted_all_raw.csv', index=False)


# find overlap between secretory data and hemolytic data, and delete it from the secretory data
df_secreted_all = pd.read_csv('data/secreted_all_raw.csv')
df_rnn = pd.read_csv('data/rnnamp/rnnamp.csv')
df_hlp = pd.read_csv('data/hlppredfuse/hlppredfuse.csv')
df_hem = pd.read_csv('data/hemolythic/hemolythic.csv')

all_dfs = [df_rnn, df_hlp, df_hem]

amp_seqs = []
for df in all_dfs:
    amp_seqs.append(list(np.array(df['text'])))
each_overlap_counter = [0] * len(all_dfs)

non_overlap_idx = []
for i in range(len(df_secreted_all)):
    current_seq = df_secreted_all.iloc[i]['text']
    dummy = True
    for df_counter in range(len(amp_seqs)):
        df_seqs = amp_seqs[df_counter]
        if current_seq in df_seqs:
            each_overlap_counter[df_counter] += 1
            dummy = False
    non_overlap_idx.append(dummy)
    if i % 10000 == 0:
        print(i)
df_secreted_all_cleaned = df_secreted_all[non_overlap_idx]


df_secreted_all_cleaned_shuffled = df_secreted_all_cleaned.sample(frac=1, random_state=42)
df_secreted_all_cleaned_shuffled.to_csv('data/swissprot/secreted_all_cleaned_all.csv', header=True, index=False)
df_secreted_all_cleaned_shuffled_train, df_secreted_all_cleaned_shuffled_test = train_test_split(df_secreted_all_cleaned_shuffled, test_size=0.1, random_state=42)
df_secreted_all_cleaned_shuffled_train = df_secreted_all_cleaned_shuffled_train[['text', 'labels']]
df_secreted_all_cleaned_shuffled_test = df_secreted_all_cleaned_shuffled_test[['text', 'labels']]

df_secreted_all_cleaned_shuffled_train.to_csv('data/swissprot/secreted_all_cleaned_all_train.csv', header=True, index=False)
df_secreted_all_cleaned_shuffled_test.to_csv('data/swissprot/secreted_all_cleaned_all_test.csv', header=True, index=False)


for dirs in ['data/swissprot/secreted_all_cleaned_all_train.csv',
             'data/swissprot/secreted_all_cleaned_all_test_cdhit.csv',
'data/hemolythic/hemolythic_train.csv',
'data/hemolythic/hemolythic_test.csv',
'data/hlppredfuse/hlppredfuse_train.csv',
'data/hlppredfuse/hlppredfuse_test.csv',
'data/rnnamp/rnnamp_train.csv',
'data/rnnamp/rnnamp_test.csv'
             ]:
    df = pd.read_csv(dirs)
    print(dirs, len(df), len(df[df['labels'] == 1]), len(df[df['labels'] == 0]))

# save as fasta for cd-hit analysis
df_secreted_all_cleaned_shuffled_train = pd.read_csv('data/swissprot/secreted_all_cleaned_all_train.csv')
df_secreted_all_cleaned_shuffled_test = pd.read_csv('data/swissprot/secreted_all_cleaned_all_test.csv')

train_seqs = [s.replace(" ", "") for s in list(df_secreted_all_cleaned_shuffled_train['text'])]
test_seqs = [s.replace(" ", "") for s in list(df_secreted_all_cleaned_shuffled_test['text'])]

train_fasta = []
test_fasta = []

for i in range(len(train_seqs)):
    train_fasta.append('>seq'+str(i)+'\n')
    train_fasta.append(train_seqs[i]+'\n')
for i in range(len(test_seqs)):
    test_fasta.append('>seq'+str(i)+'\n')
    test_fasta.append(test_seqs[i]+'\n')

with open('data/swissprot/secreted_all_cleaned_all_train.fasta', 'w+') as f:
    for line in train_fasta:
        f.write(line)
with open('data/swissprot/secreted_all_cleaned_all_test.fasta', 'w+') as f:
    for line in test_fasta:
        f.write(line)

test_cdhit = []
with open('data/swissprot/cdhit_results', 'r') as f:
    for line in f:
        test_cdhit.append(line)

remaining_idx = []
for i in test_cdhit:
    if '>' in i:
        remaining_idx.append(int(i.lstrip('>seq').rstrip('\n')))

remaining_test_df = df_secreted_all_cleaned_shuffled_test.iloc[remaining_idx]
remaining_test_df.to_csv('data/swissprot/secreted_all_cleaned_all_test_cdhit.csv', header=True, index=False)
