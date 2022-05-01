import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# Extract AMP sequences and keywords from the peptide data
interested_kw = [
    'Antimicrobial',
    'Antibiotic',
    'Antiviral defense',
    'Antiviral protein',
    'Fungicide',
    'Tumor suppressor',
    'Antiviral protein',
    'Plant defense',
    'Amphibian defense peptide',
    'Defensin',
    'Bacteriocin',
    'Lantibiotic',
    'Innate Immunity'
]
interesting_idx = []
for i in range(len(df_pep)):
    current_keywords = df_pep.iloc[i]['keyword'].split(',')
    dummy = False
    for k in current_keywords:
        if k in interested_kw:
            dummy = True
    interesting_idx.append(dummy)
df_interesting = df_pep[interesting_idx]
print(len(df), len(df_pep), len(df_interesting))
df_interesting.rename(columns={'sequence': 'text'}, inplace=True)
spaced_seqs_interesting = []
for line in df_interesting['text']:
    d = ' '.join(line)
    spaced_seqs_interesting.append(d.strip())
df_interesting['text'] = spaced_seqs_interesting
df_interesting = df_interesting.sample(frac=1, random_state=42)
df_interesting.to_csv('data/swissprot/amp_peps.csv', index=False)

keywords_conc = []
for i in range(len(df_interesting)):
    keywords_conc.extend(df_interesting.iloc[i]['keyword'].split(','))
kw_counter = {}
counter = 0
for k in keywords_conc:
    if k in kw_counter:
        kw_counter[k] += 1
    else:
        kw_counter[k] = 1
    if counter % 100000 == 0:
        print(counter)
    counter += 1

kw_counter_sorted = {k: v for k, v in sorted(kw_counter.items(), key=lambda item: item[1])}

all_kw = list(kw_counter_sorted.keys())
all_occur = list(kw_counter_sorted.values())
all_kw.reverse()
all_occur.reverse()

k= 10
top_occur = all_occur[:k]
top_kw = all_kw[:k]
top_kw.reverse()
top_occur.reverse()

colors = []
for i in range(k):
    if top_kw[i] == 'Secreted':
        colors.append('royalblue')
    else:
        colors.append('lightsteelblue')
plt.barh(range(k), top_occur, color=colors)
plt.xticks(fontsize=12)
plt.yticks(range(k), top_kw, rotation=0, fontsize=12)
plt.ylabel('UniProt Keyword', fontsize=15)
plt.xlabel('Occurrences in Antimicrobial Peptides', fontsize=12)
plt.tight_layout()
plt.savefig('results/amp_top_keywords.png', dpi=300, format='png')
plt.show()

# Extract hemolytic sequences and keywords
hem_kw = [
     'Hemolysis'
     ]
hem_idx = []
for i in range(len(df_pep)):
    current_keywords = df_pep.iloc[i]['keyword'].split(',')
    dummy = False
    for k in current_keywords:
        if k in hem_kw:
            dummy = True
    hem_idx.append(dummy)
df_hem = df_pep[hem_idx]
print(len(df), len(df_pep), len(df_hem))
df_hem.rename(columns={'sequence': 'text'}, inplace=True)
spaced_seqs_hem = []
for line in df_hem['text']:
    d = ' '.join(line)
    spaced_seqs_hem.append(d.strip())
df_hem['text'] = spaced_seqs_hem
df_hem = df_hem.sample(frac=1, random_state=42)
df_hem.to_csv('data/swissprot/hem_peps.csv', index=False)

keywords_conc = []
for i in range(len(df_hem)):
    keywords_conc.extend(df_hem.iloc[i]['keyword'].split(','))
kw_counter = {}
counter = 0
for k in keywords_conc:
    if k in kw_counter:
        kw_counter[k] += 1
    else:
        kw_counter[k] = 1
    if counter % 100000 == 0:
        print(counter)
    counter += 1

kw_counter_sorted = {k: v for k, v in sorted(kw_counter.items(), key=lambda item: item[1])}

all_kw = list(kw_counter_sorted.keys())
all_occur = list(kw_counter_sorted.values())
all_kw.reverse()
all_occur.reverse()

k= 10
top_occur = all_occur[:k]
top_kw = all_kw[:k]
top_kw.reverse()
top_occur.reverse()

colors = []
for i in range(k):
    if top_kw[i] == 'Secreted':
        colors.append('royalblue')
    else:
        colors.append('lightsteelblue')
plt.barh(range(k), top_occur, color=colors)
plt.xticks(fontsize=12)
plt.yticks(range(k), top_kw, rotation=0, fontsize=12)
plt.ylabel('UniProt Keyword', fontsize=15)
plt.xlabel('Occurrences in Hemolytic Peptides', fontsize=12)
plt.tight_layout()
plt.savefig('results/hemolysis_top_keywords.png', dpi=300, format='png')
plt.show()


