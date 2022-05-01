import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import load_dataset
from transformers import BertModel, BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig, EarlyStoppingCallback
from transformers.models.bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import IntervalStrategy
from datasets import load_metric
from datasets import load_from_disk
import random
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
import shutil
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# A new class of BERT model with different pooling mechanisms and different classification layers at the end
class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # add a hidden layer between pooler and the output
        if config.intermediate_hidden_size != 0:
            self.intermediate_classifier = nn.Linear(config.hidden_size, config.intermediate_hidden_size)
            self.classifier = nn.Linear(config.intermediate_hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # use BERT pooler
        if self.config.use_pooler == 1:
            pooled_output = outputs[1]
        # use Mean Pooling
        elif self.config.use_mean == 1:
            pooled_output = torch.sum(outputs[0] * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        # use first token pooling
        else:
            pooled_output = outputs[0][:, 0]

        pooled_output = self.dropout(pooled_output)
        # add a hidden layer between pooler and the output
        if config.intermediate_hidden_size != 0:
            intermediate_output = self.intermediate_classifier(pooled_output)
            logits = self.classifier(intermediate_output)
        else:
            logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def tokenize_function(example):
    return tokenizer(example["text"], add_special_tokens=True, truncation=True, max_length=512)


def compute_metrics(eval_preds):
    metric = load_metric("matthews_correlation")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


"""".................................CONTROL PANEL........................................"""
# choose the type of model to use, Prot-BERT which is pretrained on proteins, or normal BERT
model_type = "Rostlab/prot_bert_bfd"
do_lower_case = False

# model_type = "bert-base-uncased"
# do_lower_case = True

tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=do_lower_case)
config = AutoConfig.from_pretrained(model_type)

# Choose the subject for training
# 1 - Train on secretion detection:
# subject = 'secreted'
# 2 - Train on XGBC-Hem benchmark:
subject = 'hemolythic'
# 3 - Train on HLPPredFuse benchmark:
# subject = 'hlppredfuse'
# 4 - Train on Rnn-Hem benchmark:
# subject = 'rnnamp'

config.classifier_dropout = 0
config.hidden_dropout_prob = 0

# Choose the size of the hidden layer for the classification layer, 0 means no hidden layer
# config.intermediate_hidden_size = 1024
# config.intermediate_hidden_size = 128
config.intermediate_hidden_size = 32
# config.intermediate_hidden_size = 0


num_epochs = 50

# Whether or not to use BERT pooling (use_pooler = 1, use_mean=0), Mean pooling (use_pooler = 0,use_mean = 1),
# or first token pooling (use_pooler = 0, use_mean=0)
config.use_pooler = 0
config.use_mean = 1

# Select which layers and parameters to freeze, for selective fine-tuning only non-positional and attention are frozen
freeze_positional = 0
freeze_non_positional = 1
freeze_attention = 1
freeze_layer_norm = 0
freeze_pooler = 0

# whether to transfer from secretion detecction model
transfer = 0
# random initialization (for ablation studies)
random_init = 0
if random_init:
    transfer = 0
if subject == 'secreted':
    transfer = 0

early_stopping = 1
patience = 10
# Since all benchmarks have only train and test sets, we need to create validation sets for early stopping
if early_stopping:
    create_validation_split = 1
else:
    create_validation_split = 0

# select initial learning rate
initial_lr = 5e-4

# Lower the batch_size if you run into problem, through gradient accumulation batch_size always remains 32
batch_size = 16
accumulated_batch_size = 32

# Reload the preprocessed data
reload = False
""""............................END OF CONTROL PANEL........................................"""

model = None
dataset = None
if transfer == 0:
    if random_init:
        model = MyBertForSequenceClassification(config=config)
    else:
        model = MyBertForSequenceClassification.from_pretrained(model_type, config=config)
if transfer == 1:
    # find the secretory model with correct configuration and load it
    results_df = pd.read_csv('results/training_results.csv')
    results_df = results_df[results_df['subject'] == 'secreted']
    results_df = results_df[results_df['hidden_layer_size'] == config.intermediate_hidden_size]
    results_df = results_df[results_df['usemean'] == config.use_mean]
    results_df = results_df[results_df['usepooler'] == config.use_pooler]
    secreted_model_dir = results_df.iloc[0]['save_dir']
    model = MyBertForSequenceClassification.from_pretrained(secreted_model_dir, config=config)


# CSV file loader, 2 columns, 'text' and 'labels'. Text is spaced capital sequences.
if subject == 'secreted':
    dataset = load_dataset('csv', data_files={'train': 'data/swissprot/secreted_all_cleaned_all_train.csv', 'test': 'data/swissprot/secreted_all_cleaned_all_test.csv'},
                           cache_dir='data/processed_datasets',
                           delimiter=',',
                           )
# XGBC-Hem Benchmark
elif subject == 'hemolythic':
    dataset = load_dataset('csv', data_files={'train': 'data/hemolythic/hemolythic_train.csv',
                                              'test': 'data/hemolythic/hemolythic_test.csv'},
                           cache_dir='data/processed_datasets',
                           delimiter=',',
                           )
# HLPPredFuse Benchmark
elif subject == 'hlppredfuse':
    dataset = load_dataset('csv', data_files={'train': 'data/hlppredfuse/hlppredfuse_train.csv',
                                              'test': 'data/hlppredfuse/hlppredfuse_test.csv'},
                           cache_dir='data/processed_datasets',
                           delimiter=',',
                           )
# RNN-Hem Benchmark
elif subject == 'rnnamp':
    dataset = load_dataset('csv', data_files={'train': 'data/rnnamp/rnnamp_train.csv',
                                              'test': 'data/rnnamp/rnnamp_test.csv'},
                           cache_dir='data/processed_datasets',
                           delimiter=',',
                           )

if not reload:
    # Find the indices for train, validation, and test splits
    random.seed(42)
    data_len = len(dataset['train'])
    randomlist = list(range(data_len))
    random.shuffle(randomlist)
    valid_begin = 0
    valid_end = int(0.1 * data_len)
    valid_indices = randomlist[valid_begin:valid_end]

    # Create the validation split
    dataset['validation'] = dataset['train'].select(valid_indices)
    if create_validation_split:
        train_indices = [ind for ind in randomlist if ind not in valid_indices]
        dataset['train'] = dataset['train'].select(train_indices)

    # tokenize the data
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch", columns=['input_ids'])
    tokenized_datasets.save_to_disk('data/processed_datasets/'+subject)

else:
    tokenized_datasets = load_from_disk('data/processed_datasets/'+subject)

# remove extra columns, all that is needed are the tokenized sequence and the labels
tokenized_datasets = tokenized_datasets.remove_columns('text')
for c in tokenized_datasets.column_names['train']:
    if c in ['keyword', 'length']:
        tokenized_datasets = tokenized_datasets.remove_columns(c)

tokenized_datasets.set_format("torch")
print(tokenized_datasets['train'][1])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# save the model in a folder named via a timestamp
timestr = time.strftime("%m%d-%H%M%S")
save_dir = 'models/' + timestr + '/'
while os.path.isdir(save_dir):
    timestr = timestr.split('-')[0] + '-' + timestr.split('-')[1][:4] + str(int(timestr.split('-')[1][4:] + random.randint(1,60)))
    save_dir = 'models/' + timestr + '/'
os.makedirs(save_dir, exist_ok=True)


training_args = TrainingArguments(num_train_epochs=num_epochs,output_dir=save_dir,
                                  per_device_train_batch_size=batch_size,
                                  learning_rate=initial_lr,
                                  load_best_model_at_end=True,
                                  evaluation_strategy=IntervalStrategy.EPOCH,
                                  metric_for_best_model='eval_matthews_correlation', # monitor MCC
                                  save_total_limit=patience+1, # save minimum possible checkpoints
                                  # # prediction_loss_only=True,
                                  # use gradient accumulation to increase the effective batchsize and use less memory
                                  gradient_accumulation_steps=int(accumulated_batch_size/batch_size),
                                  eval_accumulation_steps=int(accumulated_batch_size/batch_size),
                                  # fp16=True, fp16_full_eval=True,
                                  per_device_eval_batch_size=batch_size,
                                  # # debug="underflow_overflow"
                                  )

# find all parameters of the model
param_names = []
for name, param in model.named_parameters():
    param_names.append(name)

# divide the parameters based on their type
positional_embedding_params = ['bert.embeddings.position_embeddings.weight']
non_positional_embedding_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight']
pooler_params = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
classifier_params = ['intermediate_classifier.weight', 'intermediate_classifier.bias', 'classifier.weight', 'classifier.bias']

# find all layer norm parameters
layer_norm_params = []
attention_params = []
for l in param_names:
    if 'LayerNorm' in l:
        layer_norm_params.append(l)
    elif l not in positional_embedding_params+non_positional_embedding_params+pooler_params+classifier_params:
        attention_params.append(l)
print(len(positional_embedding_params+non_positional_embedding_params+layer_norm_params+attention_params+pooler_params+classifier_params), len(param_names))

# find parameters that are not frozen
unfrozen_params = []
unfrozen_params += classifier_params
if freeze_positional == 0:
    unfrozen_params += positional_embedding_params
if freeze_non_positional == 0:
    unfrozen_params += non_positional_embedding_params
if freeze_layer_norm == 0:
    unfrozen_params += layer_norm_params
if freeze_pooler == 0:
    unfrozen_params += pooler_params
if freeze_attention == 0:
    unfrozen_params += attention_params

# freeze the parameters that need to be frozen, controlled by the control panel at the beginning of the code
frozen_counter = 0
grad_counter = 0
for name, param in model.named_parameters():
    if name in unfrozen_params:
        param.requires_grad = True
        grad_counter += len(param.flatten())
    else:
        param.requires_grad = False
        frozen_counter += len(param.flatten())

print('Frozen parameters:', frozen_counter, grad_counter, grad_counter+frozen_counter, grad_counter*100/(grad_counter+frozen_counter))

# add early stopping
callbacks = []
if early_stopping == 1:
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

trainer = None
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

trainer.train()

# DELETE the checkpoints, uncomment if space is needed
# dirs = [x[0] for x in os.walk(save_dir)]
# for d in dirs:
#     if 'checkpoint' in d:
#         shutil.rmtree(d, ignore_errors='True')

trainer.save_model()

print(trainer.evaluate())

# Inference
all_predictions_train = trainer.predict(test_dataset=tokenized_datasets['train'])
all_labels_train = np.array(all_predictions_train[1])
all_logits_train = all_predictions_train[0]
all_predictions_train = np.argmax(np.array(all_predictions_train[0]), axis=-1)

all_predictions_eval = trainer.predict(test_dataset=tokenized_datasets['validation'])
all_labels_eval = np.array(all_predictions_eval[1])
all_logits_eval = all_predictions_eval[0]
all_predictions_eval = np.argmax(np.array(all_predictions_eval[0]), axis=-1)

all_predictions_test = trainer.predict(test_dataset=tokenized_datasets['test'])
all_labels_test = np.array(all_predictions_test[1])
all_logits_test = all_predictions_test[0]
all_predictions_test = np.argmax(np.array(all_predictions_test[0]), axis=-1)
prediction_df = pd.DataFrame(data=[[all_labels_train, all_predictions_train, all_logits_train,
                                   all_labels_eval, all_predictions_eval, all_logits_eval,
                                   all_labels_test, all_predictions_test, all_logits_test]], columns=[
    'train_labels', 'train_predictions', 'train_logits',
    'eval_labels', 'eval_predictions', 'eval_logits',
    'test_labels', 'test_predictions', 'test_logits',
])
prediction_df.to_csv(save_dir+'predictions.csv', index=False)
print(save_dir)

# Print results
print('Acc:', accuracy_score(all_labels_eval, all_predictions_eval),
      ', Recall:', recall_score(all_labels_eval, all_predictions_eval), ', Precision:', precision_score(all_labels_eval, all_predictions_eval),
      ', ROC:', roc_auc_score(all_labels_eval, all_logits_eval[:,1]), ', MCC:', matthews_corrcoef(all_labels_eval, all_predictions_eval),
       ', AP:', average_precision_score(all_labels_eval, all_logits_eval[:,1]),
      ', F1:', f1_score(all_labels_eval, all_predictions_eval))

print('Acc:', accuracy_score(all_labels_test, all_predictions_test),
      ', Recall:', recall_score(all_labels_test, all_predictions_test), ', Precision:', precision_score(all_labels_test, all_predictions_test),
      ', ROC:', roc_auc_score(all_labels_test, all_logits_test[:,1]), ', MCC:', matthews_corrcoef(all_labels_test, all_predictions_test),
       ', AP:', average_precision_score(all_labels_test, all_logits_test[:,1]),
      ', F1:', f1_score(all_labels_test, all_predictions_test))

print(round(accuracy_score(all_labels_test, all_predictions_test), 4),
     round(recall_score(all_labels_test, all_predictions_test), 4), round(precision_score(all_labels_test, all_predictions_test), 4),
      round(roc_auc_score(all_labels_test, all_logits_test[:,1]), 4), round(matthews_corrcoef(all_labels_test, all_predictions_test), 4))

# Save all the empirical results to file
df = pd.DataFrame(data=[[f1_score(all_labels_eval, all_predictions_eval), accuracy_score(all_labels_eval, all_predictions_eval),
                   matthews_corrcoef(all_labels_eval, all_predictions_eval), roc_auc_score(all_labels_eval, all_logits_eval[:,1]),
                   precision_score(all_labels_eval, all_predictions_eval),
                   recall_score(all_labels_eval, all_predictions_eval), average_precision_score(all_labels_eval, all_logits_eval[:,1]),
                   f1_score(all_labels_test, all_predictions_test), accuracy_score(all_labels_test, all_predictions_test),
                   matthews_corrcoef(all_labels_test, all_predictions_test), roc_auc_score(all_labels_test, all_logits_test[:,1]),
                   precision_score(all_labels_test, all_predictions_test),
                   recall_score(all_labels_test, all_predictions_test), average_precision_score(all_labels_test, all_logits_test[:,1]),
                   subject, config.use_pooler, config.use_mean, config.intermediate_hidden_size, batch_size,
                   freeze_positional, patience, freeze_non_positional, freeze_attention,
                   freeze_layer_norm, freeze_pooler, initial_lr, transfer, num_epochs, save_dir, early_stopping, create_validation_split]], columns=[
                   'f1_eval', 'accuracy_eval', 'mcc_eval', 'roc_auc_eval', 'precision_eval', 'recall_eval', 'average_precision_score_eval',
                   'f1_test', 'accuracy_test', 'mcc_test', 'roc_auc_test', 'precision_test', 'recall_test', 'average_precision_score_test',
                   'subject', 'usepooler', 'usemean', 'hidden_layer_size', 'batchsize', 'frozenpositional', 'patience',
                   'frozennonpositional', 'frozenattention', 'frozenlayernorm', 'frozenpooler', 'lr', 'transfer', 'num_epochs', 'save_dir',
                   'early_stopping', 'create_validation_split'])
if not os.path.isfile('results/training_results.csv'):
   df.to_csv('results/training_results.csv', index=False)
else: # else it exists so append without writing the header
   df.to_csv('results/training_results.csv', index=False, mode='a', header=False)
