# AMPDeep: Hemolytic Activity Prediction of Antimicrobial Peptides using Transfer Learning
Deep learning's automatic feature extraction has proven to give superior performance in many sequence classification tasks. However, deep learning models generally require a massive amount of data to train, which in the case of Hemolytic Activity Prediction of Antimicrobial Peptides creates a challenge due to the small amount of available data.

In this work transfer learning is leveraged to overcome the challenge of small data and a deep learning based model is  successfully adopted for hemolysis activity classification of antimicrobial peptides. This model is first initialized as a protein language model which is pre-trained on masked amino acid prediction on many unlabeled protein sequences in a self-supervised manner. Having done so, the model is fine-tuned on an aggregated dataset of labeled peptides in a supervised manner to predict secretion. Through transfer learning, hyper-parameter optimization and selective fine-tuning, AMPDeep is able to achieve state-of-the-art performance on three hemolysis benchmarks using only the sequence of the peptides. This work assists the adoption of large sequence-based models for peptide classification and modeling tasks in a practical manner. 

Three different benchmarks for hemolysis activity prediction of therapeutic and antimicrobial peptides are gathered and the AMPDeep pipeline is implemented for each. The result demonstrate that AMPDeep outperforms the previous works on all three benchmarks, including works that use physicochemical features to represent the peptides or those who solely rely on the sequence and use deep learning to learn representation for the peptides. AMPDeep fine-tunes a large transformer based model on a small amount of peptides and successfully leverages the patterns learned from other protein and peptide databases to assist hemolysis activity prediction modeling. 

# Requirements
transformers==4.8.1
rdkit==2019.09.03
torch==1.9.0
pandas==1.1.5
numpy
scikit-learn
matplotlib

# UniProt Keyword Analysis (optional)
To find the most common keywords associated with antimicrobial peptides (AMPs) and hemolytic peptides, the swissprot dataset is needed. First download the swissprot dataset and copy it to the 'data/swissprot' directory. Having done so, the keyword_analysis.py script can find the sequences and keywords for each entry, and count the instances where each keyword is used. In the case of AMPs and hemolytic peptides, "Secretion" was one of the most frequent keywords, demonstrating its high association with our data of interest and its potential for being a source task for transfer learning.

# Data Preprocessing (optional)
Three different benchmarks are used in this work to evaluate the performance of AMPDeep for hemolysis activity prediction. The data for each benchmark is preprocessed in the preprocessing_hemolysis.py script, through adding spaces between each amino acid token of the sequences, then shuffling and saving the training and the independent test splits. In order to create a dataset from secretory and non-secretory peptide, the preprocess_secretion.py script is used. This process finds secretory peptides within the Swissprot database using "Secretion" keyword, and finds the non-secretory peptides through following the opposite scenario, however adding the constraint that the peptides must be located within the Cytoplasm. The secretory benchmark as well as the three hemolytic activty prediction benchmark can be foundd within the 'data' directory.

# Training and Hyper-Parameter Optimization
The training.py script offers a control panel that controls all parameters needed for training such as epoch number, learning rate, and early stopping. Moreover, through this control panel different sections of the model can be frozen during training, or the pooling mechanism of the model can be changed to alternate between BERT pooling, mean pooling, and first token poolig. To perform training on the secretion task or any of the emolysis benchmarks, change the 'subject' parameter within this control panel. Selective fine-tuning happens when the non-positional embeddings as well as the attention heads are frozen, while the positional embedding, the layer norm parameters, and the classification parameters are fine-tuned. 
