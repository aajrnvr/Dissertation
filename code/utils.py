# This file is utils, which contains useful functions used in other .py files

from openprompt.data_utils import InputExample
import torch
import pandas as pd
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from openprompt.utils.logging import logger
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchnlp.encoders import LabelEncoder
from funcsigs import signature
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report

def vote_score(df, score, location):
    '''
    Inputs: df: DataFrame based on clinical notes; columns ['ID', 'Label', ...]
            score: prediction values based on clinical notes
            location: where roc curve will be saved

    Action: plot roc curve and save to location

    Output: fpr: false positive rate
            tpr: true positive rate
            df_out: DataFrame after grouping by patient ID
    '''
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    df_out = pd.DataFrame({'logits': temp.values, 'label': x})
    # df_out = pd.DataFrame({'logits': temp.values, 'ID': x}) # Previously x has been wrongly assign a column name ID, but its actual meaning is true label

    # print(temp.values)
    fpr, tpr, thresholds = roc_curve(x, temp.values)
    auc_score = auc(fpr, tpr)

    plt.clf() # New
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    string = os.path.join(location,'auroc_clinicalbert_Prompt'+'.png')
    plt.savefig(os.path.join('/home/s2174572/mlp/mlp1/result_all_prompt', string))

    return fpr, tpr, df_out

def vote_score_no_plot(df, score, location):
    '''
    Inputs: df: DataFrame based on clinical notes; columns ['ID', 'Label', ...]
            score: prediction values based on clinical notes
            location: where roc curve will be saved

    Action: plot roc curve and save to location

    Output: fpr: false positive rate
            tpr: true positive rate
            df_out: DataFrame after grouping by patient ID
    '''
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    df_out = pd.DataFrame({'logits': temp.values, 'label': x})
    # df_out = pd.DataFrame({'logits': temp.values, 'ID': x}) # Previously x has been wrongly assign a column name ID, but its actual meaning is true label

    # print(temp.values)
    fpr, tpr, thresholds = roc_curve(x, temp.values)
    # auc_score = auc(fpr, tpr)

    return fpr, tpr, df_out

def pr_curve_plot(y, y_score, location):

    """
    Input: y: label (after grouping by patient ID)
           y_score: prediction value (after grouping by patient ID)
           location: where plot will be saved

    Action: plot prc and save to location

    Output: None 

    """
    precision, recall, _ = precision_recall_curve(y, y_score)
    area = auc(recall,precision)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.clf() # New
    plt.figure(2)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
              area))
    
    string = os.path.join(location,'auprc_clinicalbert_Prompt'+'.png')
    plt.savefig(os.path.join('/home/s2174572/mlp/mlp1/result_all_prompt', string))


def vote_pr_curve(df, score,location):

    '''
    Input: df: DataFrame based on clinical notes; columns ['ID', 'Label', ...]
           score: prediction values based on clinical notes
           location: where prc curve will be saved
    Action: plot prc curve and save to location
    Output: rp80 (after grouping by patient ID)
    '''

    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    y = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    
    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
    
    pr_curve_plot(y, temp, location)
    
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    
    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80

def summary_from_df(df):
    '''
    Input: df: DataFrame
           
    Output: true_label, pred_label: 'label' and 'pred_class' column in df
            others: summary statistics
    '''
    # df = pd.read_csv(result_path)
    true_label = df['label']
    pred_label = df['pred_class'] # 1.0*(df['logits'] > 0.5)

    pred_mean = df['logits'].mean()
    pred_var = df['logits'].var()
    df_grouped_mean = df.groupby(['label']).mean()
    df_grouped_var = df.groupby(['label']).var()

    pred_posi_mean = df_grouped_mean['logits'][1]
    pred_neg_mean = df_grouped_mean['logits'][0]

    pred_posi_var = df_grouped_var['logits'][1]
    pred_neg_var = df_grouped_var['logits'][0]

    return true_label, pred_label, pred_mean, pred_var, pred_posi_mean, pred_posi_var, pred_neg_mean, pred_neg_var


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import itertools
# Usually choose this: class_names = ['not r/a', 'r/a']
def plot_confusion_matrix(cm, class_names, save_dir = None):
    '''
    Input: cm: confusion matrix
           class_names: class names on the plot
           save_dir: path where plot is saved
    Action: Returns a matplotlib figure containing the plotted confusion matrix.
    '''
    """
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')
    

    figure = plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # Use white text if squares are dark; otherwise black.
    # threshold = cm.max() / 2.
    threshold = 0.5
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    figure.savefig(f'{save_dir}/test_mtx.png')
    
    return figure


def Group_and_Sample(mimic_data_dir, num_examples_per_label=32, seed=1, location = 'result_all_prompt/prompt_pipeline'):
    '''
    first read train/val/test from mimic_data_dir, group them by ID then sample for each label,
    save the modified data in csv and return the folder path where modified csv file has been created
    '''
    import random
    random.seed(seed)
    mimic_data_modified_dir = f'/home/s2174572/mlp/mlp1/{location}/few_shot_data'
    for f in ['train.csv','val.csv','test.csv']:
        df = pd.read_csv(f'{mimic_data_dir}/{f}')
        df_sort = df.sort_values(by=['ID'])
        x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
        IDs = df_sort['ID'].unique() # list
        df_out = pd.DataFrame({'ID': IDs, 'label': x}) # after grouping by petient ID

        ID_list_0 = list(df_out[df_out['label']==0.0]['ID']) # IDs whose label is 0
        ID_list_1 = list(df_out[df_out['label']==1.0]['ID']) # IDs whose label is 1
        sample_IDs_0 = random.sample(ID_list_0, num_examples_per_label)
        sample_IDs_1 = random.sample(ID_list_1, num_examples_per_label)
        sample_IDs = sample_IDs_0 + sample_IDs_1 # combine them together, this is what the IDs we should choose
        df_sampled = df[df['ID'].isin(sample_IDs)]
        df_sampled = df_sampled[['ID','TEXT','Label']]
        df_sampled.to_csv(f'{mimic_data_modified_dir}/{f}')

    return mimic_data_modified_dir

def label_words_from_statistics(train_dir, K, current_label_words_address, list0=None, list1=None):
    '''
    args: 
    train_dir: csv document dir where we want to get label words from
    k: top k common words, NOTE this number is before set abstracion
    current_label_words_address: where txt will be saved
    list0: predefined label words for class 0
    list1: predefined label words for class 1

    output: .txt address
    '''
    import collections
    import os
    import string
    import pandas as pd 
    import numpy as np
    data0='' # clinical notes with label 0.0
    data1='' # clinical notes with label 1.0
    with open(train_dir) as f:
        for line in f.readlines():
            line = line.strip()
            if line.endswith(',0.0'):
                # omit last four chars ,0.0
                data0 += line[:-4]
            if line.endswith(',1.0'):
                data1 += line[:-4]
    f.close()

    data0 = data0.translate(str.maketrans('', '', string.punctuation)) # remove punctuations in original context
    data0 = data0.split(' ') # list

    data1 = data1.translate(str.maketrans('', '', string.punctuation))
    data1 = data1.split(' ') # list

    counter0 = collections.Counter(data0) # Store in a dictionary
    counter1 = collections.Counter(data1) # Store in a dictionary

    # Get top 200 words from clinical note (label 0 and label 1)
    set0 = set()
    for i in np.arange(K):
        set0.add(counter0.most_common(K)[i][0])

    set1 = set()
    for i in np.arange(K):
        set1.add(counter1.most_common(K)[i][0])

    set0_1 = set0 - set1 # words in clinical notes with label 0 but not in label 1 
    set1_0 = set1 - set0 # words in clinical notes with label 1 but not in label 0 

    if list0 != None:
        for word in list0:
            set0_1.add(word)
    
    if list1 != None:
        for word in list1:
            set1_0.add(word)
    
    new_f = open(current_label_words_address,'w')
    for i, element in enumerate(list(set0_1)):
        if i < len(list(set0_1)) - 1:
            new_f.write(element+', ')
        elif i == len(list(set0_1)) - 1:
            new_f.write(element)

    new_f.write('\n')

    for j, element in enumerate(list(set1_0)):
        if j < len(list(set1_0)) - 1:
            new_f.write(element+', ')
        elif j == len(list(set1_0)) - 1:
            new_f.write(element)

    new_f.close()


    return current_label_words_address




class MimicProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()        

    def get_examples(self, data_dir, set = "train"):
        path = f"{data_dir}/{set}.csv"
        print(f"loading {set} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)
        self.label_encoder = LabelEncoder(np.unique(df.Label).tolist(), reserved_labels = [])
        
        for idx, row in tqdm(df.iterrows()):
            # print(row)
            _, ID, body, label = row
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=int(ID), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!")      
        return examples

def summarize_predictions(df_out, location):
    '''Note df_out is a pd.dataframe and MUST come fron utils.vote_score'''

    # mean and variance for all prediction values
    pred_mean = df_out['logits'].mean()
    pred_var = df_out['logits'].var()
    # mean and variance for prediction values of posituve labelled notes
    df2 = df_out.groupby(['label']).mean()
    df3 = df_out.groupby(['label']).var()
    posi_pred_mean = df2['logits'][1]
    posi_pred_var = df3['logits'][1]
    # mean and variance for prediction values of negative labelled notes
    neg_pred_mean = df2['logits'][0]
    neg_pred_var = df3['logits'][0]

    df_out.to_csv(f"/home/s2174572/mlp/mlp1/result_all_prompt/{location}/result.csv")
    # rp80, we still use notation from vote_pr_curve functionn
    temp = df_out['logits']
    y = df_out['label']
    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
    
    pr_curve_plot(y, temp, location)
    
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    
    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    # Print out prediction mean and variance for ALL labelled examples
    print('Here we use mean and variance for ALL predictions to analyze how prediction value distributed')
    print('mean: ',pred_mean)
    print('variance: ',pred_var)

    # Print out prediction mean and variance for POSITIVE labelled examples
    print('mean and variance for POSITIVE labelled example')
    print('mean: ',posi_pred_mean)
    print('variance: ',posi_pred_var)

    # Print out prediction mean and variance for NEGATIVE labelled examples
    print('mean and variance for NEGATIVE labelled example')
    print('mean: ',neg_pred_mean)
    print('variance: ',neg_pred_var)

    # classification report
    from sklearn.metrics import f1_score
    y_pred = df_out['logits'] > 0.5 
    y_label = df_out['label']
    report = classification_report(y_label, y_pred)
    print(report)
    # print(f1_score(y_label,y_pred,average = 'binary'))

#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Here we modify original code from openprompt/data_utils/data_sampler to work with finetune for FewShot, this can actually not in use now.
from collections import defaultdict, namedtuple
from typing import *

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from openprompt.utils.logging import logger

from typing import Union




class FewShotSampler(object):
    '''
    Few-shot learning is an important scenario for prompt-learning, this is a sampler that samples few examples over each class.

    Args:
        num_examples_total(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.
        also_sample_dev(:obj:`bool`, optional): Whether to apply the sampler to the dev data.
        num_examples_total_dev(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label_dev(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.
    
    '''

    def __init__(self,
                 num_examples_total: Optional[int]=None,
                 num_examples_per_label: Optional[int]=None,
                 also_sample_dev: Optional[bool]=False,
                 num_examples_total_dev: Optional[int]=None,
                 num_examples_per_label_dev: Optional[int]=None,
                 ):
        if num_examples_total is None and num_examples_per_label is None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both None.")
        elif num_examples_total is not None and num_examples_per_label is not None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both set.")
        
        if also_sample_dev:
            if num_examples_total_dev is not None and num_examples_per_label_dev is not None:
                raise ValueError("num_examples_total and num_examples_per_label can't be both set.")
            elif num_examples_total_dev is None and num_examples_per_label_dev is None:
                logger.warning(r"specify neither num_examples_total_dev nor num_examples_per_label_dev,\
                                set to default (equal to train set setting).")
                self.num_examples_total_dev = num_examples_total
                self.num_examples_per_label_dev = num_examples_per_label
            else:
                self.num_examples_total_dev  = num_examples_total_dev
                self.num_examples_per_label_dev = num_examples_per_label_dev

        self.num_examples_total = num_examples_total
        self.num_examples_per_label = num_examples_per_label
        self.also_sample_dev = also_sample_dev

    def __call__(self, 
                 train_dataset: Union[Dataset, List],
                 valid_dataset: Optional[Union[Dataset, List]] = None,
                 seed: Optional[int] = None
                ) -> Union[Dataset, List]:
        '''
        The ``__call__`` function of the few-shot sampler.

        Args:
            train_dataset (:obj:`Union[Dataset, List]`): The train datset for the sampler.
            valid_dataset (:obj:`Union[Dataset, List]`, optional): The valid datset for the sampler. Default to None.
            seed (:obj:`int`, optional): The random seed for the sampling.
        
        Returns:
            :obj:`(Union[Dataset, List], Union[Dataset, List])`: The sampled dataset (train_dataset, valid_dataset), whose type is identical to the input.

        '''
        if valid_dataset is None:
            if self.also_sample_dev:
                return self._sample(train_dataset, seed, sample_twice=True)
            else:
                return self._sample(train_dataset, seed, sample_twice=False)
        else:
            train_dataset = self._sample(train_dataset, seed)
            if self.also_sample_dev:
                valid_dataset = self._sample(valid_dataset, seed)
            return train_dataset, valid_dataset
    
    def _sample(self, 
                data: Union[Dataset, List], 
                seed: Optional[int],
                sample_twice = False,
               ) -> Union[Dataset, List]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        indices = [i for i in range(len(data))]

        if self.num_examples_per_label is not None:
            # assert hasattr(data[0], 'label'), "sample by label requires the data has a 'label' attribute." # New!
            labels = [x[-1] for x in data] # New!
            selected_ids = self.sample_per_label(indices, labels, self.num_examples_per_label) # TODO fix: use num_examples_per_label_dev for dev
        else:
            selected_ids = self.sample_total(indices, self.num_examples_total)
        
        if sample_twice:
            selected_set = set(selected_ids)
            remain_ids = [i for i in range(len(data)) if i not in selected_set]
            if self.num_examples_per_label_dev is not None:
                assert hasattr(data[0], 'label'), "sample by label requires the data has a 'label' attribute."
                remain_labels = [x.label for idx, x in enumerate(data) if idx not in selected_set]
                selected_ids_dev = self.sample_per_label(remain_ids, remain_labels, self.num_examples_per_label_dev)
            else:
                selected_ids_dev = self.sample_total(remain_ids, self.num_examples_total_dev)
        
            if isinstance(data, Dataset):
                return Subset(data, selected_ids), Subset(data, selected_ids_dev)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids], [data[i] for i in selected_ids_dev]
        
        else:
            if isinstance(data, Dataset):
                return Subset(data, selected_ids)
            elif isinstance(data, List):
                return [data[i] for i in selected_ids]
        
    
    def sample_total(self, indices: List, num_examples_total):
        '''
        Use the total number of examples for few-shot sampling (Strategy ``I``).
        
        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            num_examples_total(:obj:`int`): The total number of examples.
        
        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
            
        '''
        self.rng.shuffle(indices)
        selected_ids = indices[:num_examples_total]
        logger.info("Selected examples (mixed) {}".format(selected_ids))
        return selected_ids

    def sample_per_label(self, indices: List, labels, num_examples_per_label):
        '''
        Use the number of examples per class for few-shot sampling (Strategy ``II``). 
        If the number of examples is not enough, a warning will pop up.
        
        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            labels(:obj:`List`): The list of the labels.
            num_examples_per_label(:obj:`int`): The total number of examples for each class.
        
        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
        '''

        ids_per_label = defaultdict(list)
        selected_ids = []
        for idx, label in zip(indices, labels):
            ids_per_label[label].append(idx)
        for label, ids in ids_per_label.items():
            tmp = np.array(ids)
            self.rng.shuffle(tmp)
            if len(tmp) < num_examples_per_label:
                logger.info("Not enough examples of label {} can be sampled".format(label))
            selected_ids.extend(tmp[:num_examples_per_label].tolist())
        selected_ids = np.array(selected_ids)
        self.rng.shuffle(selected_ids)
        selected_ids = selected_ids.tolist()    
        logger.info("Selected examples {}".format(selected_ids))
        return selected_ids
#------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# We paste soft verbalizer class from openprompt V0.1.2 here
from inspect import Parameter
import json
from os import stat
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput

from transformers.models.t5 import  T5ForConditionalGeneration

class SoftVerbalizer(Verbalizer):
    r"""
    The implementation of the verbalizer in `WARP <https://aclanthology.org/2021.acl-long.381/>`_

    Args:   
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self, 
                 tokenizer: Optional[PreTrainedTokenizer],
                 model: Optional[PreTrainedModel],
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler

        head_name = [n for n,c in model.named_children()][-1]
        logger.info(f"The LM head named {head_name} was retrieved.")
        self.head = copy.deepcopy(getattr(model, head_name))
        max_loop = 5
        if not isinstance(self.head, torch.nn.Linear):
            module = self.head
            found = False
            last_layer_full_name = []
            for i in range(max_loop):
                last_layer_name = [n for n,c in module.named_children()][-1]
                last_layer_full_name.append(last_layer_name)
                parent_module = module
                module = getattr(module, last_layer_name)
                if isinstance(module, torch.nn.Linear):
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Can't not retrieve a linear layer in {max_loop} loop from the plm.")
            self.original_head_last_layer = module.weight.data
            self.hidden_dims = self.original_head_last_layer.shape[-1]
            self.head_last_layer_full_name = ".".join(last_layer_full_name)
            self.head_last_layer = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)
            setattr(parent_module, last_layer_name, self.head_last_layer)
        else:
            self.hidden_dims = self.head.weight.shape[-1]
            self.original_head_last_layer = getattr(model, head_name).weight.data
            self.head = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)


        if label_words is not None: # use label words as an initialization
            self.label_words = label_words
        
        

        
    @property
    def group_parameters_1(self,):
        r"""Include the parameters of head's layer but not the last layer
        In soft verbalizer, note that some heads may contain modules 
        other than the final projection layer. The parameters of these part should be
        optimized (or freezed) together with the plm.
        """
        if isinstance(self.head, torch.nn.Linear):
            return []
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name not in n]
 
    @property
    def group_parameters_2(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()]
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name in n]


    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

        

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

        

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token. 
        """
        words_ids = []
        for word in self.label_words:
            if isinstance(word, list):
                logger.warning("Label word for a class is a list, only use the first word.")
            word = word[0]
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning("Word {} is split into multiple tokens: {}. \
                    If this is not what you expect, try using another word for this verbalizer" \
                    .format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)

        max_len  = max([len(ids) for ids in words_ids])
        words_ids_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in words_ids]
        words_ids = [ids+[0]*(max_len-len(ids)) for ids in words_ids]
        
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        
        init_data = self.original_head_last_layer[self.label_words_ids,:]*self.label_words_mask.to(self.original_head_last_layer.dtype).unsqueeze(-1)
        init_data = init_data.sum(dim=1)/self.label_words_mask.sum(dim=-1,keepdim=True)

        if isinstance(self.head, torch.nn.Linear):
            self.head.weight.data = init_data
            self.head.weight.data.requires_grad=True
        else:
            '''
            getattr(self.head, self.head_last_layer_full_name).weight.data = init_data
            getattr(self.head, self.head_last_layer_full_name).weight.data.requires_grad=True # To be sure
            '''
            self.head_last_layer.weight.data = init_data
            self.head_last_layer.weight.data.requires_grad=True

            

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 
        """
        label_logits = self.head(hiddens)
        return label_logits

    

    def process_outputs(self, outputs: torch.Tensor, batch: Union[Dict, InputFeatures], **kwargs):
        return self.process_hiddens(outputs)



    def gather_outputs(self, outputs: ModelOutput):
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")

        return ret
# -------------------------------------------------------------------
'''
# To deal with errors occured in LM-BFF experiments using old openprompt version

from abc import abstractmethod
from builtins import ValueError
from typing import List, Optional, Dict, Union
from tokenizers import Tokenizer
import json
import torch
import torch.nn.functional as F
from yacs.config import CfgNode
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.pipeline_base import PromptDataLoader, PromptModel

from openprompt.prompt_base import Template, Verbalizer
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from ..utils import logger
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Optional, Dict
import itertools
import numpy as np
from ..utils import signature
from ..config import convert_cfg_to_dict
from torch.nn.parallel import DataParallel

class LMBFFTemplateGenerationTemplate(ManualTemplate):
    """
    This is a special template used only for earch of template in LM-BFF. For example, a template could be ``{"placeholder": "text_a"}{"mask"}{"meta":"labelword"}{"mask"}``, where ``{"meta":"labelword"}`` is replaced by label_words in verbalizer in `wrap_one_example` method, and ``{"mask"}`` is replaced by special tokens used for generation, for T5, it is ``<extra_id_0>, <extra_id_1>, ...``.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        verbalizer (:obj:`ManualVerbalizer`): A verbalizer to provide label_words.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    def __init__(self,
                 tokenizer: T5Tokenizer,
                 verbalizer: ManualVerbalizer,
                 text: Optional[List[str]] = None,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         text = text,
                         placeholder_mapping=placeholder_mapping)
        self.verbalizer = verbalizer

    def wrap_one_example(self,
                         example: InputExample) -> List[Dict]:
        example.meta['labelword'] = self.verbalizer.label_words[example.label][0].strip()
        wrapped_example = super().wrap_one_example(example)
        return wrapped_example

class TemplateGenerator:
    r""" This is the automatic template search implementation for `LM-BFF <https://arxiv.org/pdf/2012.15723.pdf>`_. It uses a generation model to generate multi-part text to fill in the template. By jointly considering all samples in the dataset, it uses beam search decoding method to generate a designated number of templates with the highest probability. The generated template may be uniformly used for all samples in the dataset.

    Args:
        model (:obj:`PretrainedModel`): A pretrained model for generation.
        tokenizer (:obj:`PretrainedTokenizer`): A corresponding type tokenizer.
        tokenizer_wrapper (:obj:`TokenizerWrapper`): A corresponding type tokenizer wrapper class.
        max_length (:obj:`Optional[int]`): The maximum length of total generated template. Defaults to 20.
        target_number (:obj:`Optional[int]`): The number of separate parts to generate, e.g. in T5, every <extra_id_{}> token stands for one part. Defaults to 2.
        beam_width (:obj:`Optional[int]`): The beam search width.  Defaults to 100.
        length_limit (:obj:`Optional[List[int]]`): The length limit for each part of content, if None, there is no limit. If not None, the list should have a length equal to `target_number`. Defaults to None.
        forbidden_word_ids (:obj:`Optional[List[int]]`): Any tokenizer-specific token_id you want to prevent from generating. Defaults to `[]`, i.e. all tokens in the vocabulary are allowed in the generated template.
    """
    def __init__(self,
                model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper: Tokenizer,
                 verbalizer: Verbalizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None,
                 forbidden_word_ids: Optional[List[int]] = [],
                 config: CfgNode = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.verbalizer= verbalizer
        self.target_number = target_number # number of parts to generate in one sample
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_limit = length_limit
        self.probs_buffer, self.labels_buffer = None, None

        # Forbid single space token, "....", and "..........", and some other tokens based on vocab
        self.forbidden_word_ids = forbidden_word_ids
        self.sent_end_id = self.tokenizer.convert_tokens_to_ids('.')

        self.input_ids_buffer, self.attention_mask_buffer, self.labels_buffer = None, None, None

        self.config = config

    @property
    def device(self):
        r"""
        return the device of the model
        """
        if isinstance(self.model, DataParallel):
            return self.model.module.device
        else:
            return self.model.device

    def _register_buffer(self, data):
        if self.input_ids_buffer is None :
            self.input_ids_buffer = data.input_ids.detach()
            self.attention_mask_buffer = data.attention_mask.detach()
            self.labels_buffer = data.label.detach()
        else:
            self.input_ids_buffer = torch.vstack([self.input_ids_buffer, data.input_ids.detach()])
            self.attention_mask_buffer = torch.vstack([self.attention_mask_buffer, data.attention_mask.detach()])
            self.labels_buffer = torch.hstack([self.labels_buffer, data.label.detach()])

    @abstractmethod
    def get_part_token_id(self, part_id: int) -> int:
        r"""
        Get the start token id for the current part. It should be specified according to the specific model type. For T5 model, for example, the start token for `part_id=0` is `<extra_id_0>`, this method should return the corresponding token_id.
        Args:
            part_id (:obj:`int`): The current part id (starts with 0).
        Returns:
            token_id (:obj:`int`): The corresponding start token_id.
        """
        raise NotImplementedError


    def convert_template(self, generated_template: List[str], original_template: List[Dict]) -> str:
        r"""
        Given original template used for template generation,convert the generated template into a standard template for downstream prompt model, return a ``str``
        Example:
        generated_template: ['<extra_id_0>', 'it', 'is', '<extra_id_1>', 'one', '</s>']
        original_template: [{'add_prefix_space': '', 'placeholder': 'text_a'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': ' ', 'meta': 'labelword'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': '', 'text': '.'}]
        return: "{'placeholder':'text_a'} it is {"mask"} one."
        """
        i = 0
        part_id = 0
        while generated_template[i] != self.tokenizer.additional_special_tokens[part_id] and i < len(generated_template) - 1:
            i += 1
        assert generated_template[i] == self.tokenizer.additional_special_tokens[part_id], print('invalid generated_template {}, missing token {}'.format(generated_template, self.tokenizer.additional_special_tokens[part_id]))
        i += 1

        output = []
        for d in original_template:
            if 'mask' in d:
                j = i + 1
                part_id += 1
                while generated_template[j] != self.tokenizer.additional_special_tokens[part_id] and j < len(generated_template) - 1:
                    j += 1
                output.append(d.get('add_prefix_space', '') + self.tokenizer.convert_tokens_to_string(generated_template[i:j]))
                i = j + 1
            elif 'meta' in d and d['meta'] == 'labelword':
                output.append(d.get('add_prefix_space', '') + '{"mask"}')
            elif 'text' in d:
                output.append(d.get('add_prefix_space', '') + d['text'])
            else:
                prefix = d.get('add_prefix_space', '')
                if 'add_prefix_space' in d:
                    d.pop('add_prefix_space')
                output.append(prefix + json.dumps(d))
        return ''.join(output)



    def _get_templates(self):
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        input_ids = self.input_ids_buffer
        attention_mask = self.attention_mask_buffer

        ori_decoder_input_ids = torch.zeros((input_ids.size(0), self.max_length)).long()
        ori_decoder_input_ids[..., 0] = inner_model.config.decoder_start_token_id


        # decoder_input_ids: decoder inputs for next regressive generation
        # ll: log likelihood
        # output_id: which part of generated contents we are at
        # output: generated content so far
        # last_length (deprecated): how long we have generated for this part
        current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
        for i in tqdm(range(self.max_length - 2)):
            new_current_output = []
            for item in current_output:
                if item['output_id'] > self.target_number:
                    # Enough contents
                    new_current_output.append(item)
                    continue
                decoder_input_ids = item['decoder_input_ids']

                # Forward
                batch_size = 32
                turn = input_ids.size(0) // batch_size
                if input_ids.size(0) % batch_size != 0:
                    turn += 1
                aggr_output = []
                for t in range(turn):
                    start = t * batch_size
                    end = min((t + 1) * batch_size, input_ids.size(0))

                    with torch.no_grad():
                        aggr_output.append(self.model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.to(input_ids.device)[start:end])[0])
                aggr_output = torch.cat(aggr_output, 0)

                # Gather results across all input sentences, and sort generated tokens by log likelihood
                aggr_output = aggr_output.mean(0)
                log_denominator = torch.logsumexp(aggr_output[i], -1).item()
                ids = list(range(inner_model.config.vocab_size))
                ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
                ids = ids[:self.beam_width+3]

                for word_id in ids:
                    output_id = item['output_id']

                    if word_id == self.get_part_token_id(output_id) or word_id == self.tokenizer.eos_token_id:
                        # Finish one part
                        if self.length_limit is not None and item['last_length'] < self.length_limit[output_id - 1]:
                            check = False
                        else:
                            check = True
                        output_id += 1
                        last_length = 0
                    else:
                        last_length = item['last_length'] + 1
                        check = True

                    output_text = item['output'] + [word_id]
                    ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                    new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                    new_decoder_input_ids[:] = decoder_input_ids
                    new_decoder_input_ids[..., i + 1] = word_id

                    if word_id in self.forbidden_word_ids:
                        check = False

                    # Forbid continuous "."
                    if len(output_text) > 1 and output_text[-2] == self.sent_end_id and output_text[-1] == self.sent_end_id:
                        check = False

                    if check:
                        # Add new results to beam search pool
                        new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                        new_current_output.append(new_item)

            if len(new_current_output) == 0:
                break

            new_current_output.sort(key=lambda x: x['ll'], reverse=True)
            new_current_output = new_current_output[:self.beam_width]
            current_output = new_current_output

        return [self.tokenizer.convert_ids_to_tokens(item['output']) for item in current_output]

    def _show_template(self):
        logger.info("Templates are \n{}".format('\n'.join(self.templates_text)))


    @classmethod
    def from_config(cls, config: CfgNode, **kwargs,):
        r"""
        Returns:
            template_generator (:obj:`TemplateGenerator`)
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        init_dict['config'] = config
        template_generator = cls(**init_dict)
        return template_generator


    def release_memory(self):
        self.model = self.model.cpu()

    def generate(self, dataset: List[InputExample]):
        r"""
        Args:
            dataset (:obj:`List[InputExample]`): The dataset based on which template it to be generated.
        Returns:
            template_text (:obj:`List[str]`): The generated template text
        """
        template_for_auto_t = LMBFFTemplateGenerationTemplate.from_config(config=self.config.template, tokenizer=self.tokenizer, verbalizer = self.verbalizer)

        dataloader = PromptDataLoader(dataset, template_for_auto_t, self.tokenizer, self.tokenizer_wrapper, batch_size=len(dataset), decoder_max_length=128) # register all data at once
        for data in dataloader:
            data = data.to(self.device)
            self._register_buffer(data)

        self.model.eval()
        with torch.no_grad():
            self.templates_text = self._get_templates() # List[str]
            original_template = template_for_auto_t.text
            self.templates_text = [self.convert_template(template_text, original_template) for template_text in self.templates_text]
            self._show_template()
        return self.templates_text


class T5TemplateGenerator(TemplateGenerator):
    r"""
    Automatic template search using T5 model. This class inherits from ``TemplateGenerator``.
    """
    def __init__(self,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 tokenizer_wrapper: Tokenizer,
                 verbalizer: Verbalizer,
                 max_length: Optional[int] = 20,
                 target_number: Optional[int] = 2,
                 beam_width: Optional[int] = 100,
                 length_limit: Optional[List[int]] = None,
                 forbidden_word_ids: Optional[List[int]] = [3, 19794, 22354],
                 config: CfgNode = None):
        super().__init__(model = model,
                        tokenizer = tokenizer,
                        tokenizer_wrapper=tokenizer_wrapper,
                        verbalizer = verbalizer,
                        max_length = max_length,
                        target_number= target_number,
                        beam_width = beam_width,
                        length_limit = length_limit,
                        forbidden_word_ids = forbidden_word_ids,
                        config=config)

    def get_part_token_id(self, part_id):
        return self.tokenizer.additional_special_tokens_ids[part_id]


    # def convert_template(self, generate_text_list):
    #     # original_template = self.template_for_auto_t.text
    #     text_list = self.tokenizer.convert_tokens_to_string(generate_text_list).replace('<extra_id_0>', '{"placeholder":"text_a"}').replace('<extra_id_1>', ' {"mask"}').replace('<extra_id_2>', ' {"placeholder":"text_b"}').replace('</s>', '').replace('  ', ' ').split(' ')
    #     # incase no <extra_id_1> (generation stop by maximum length)
    #     if '{"mask"}' not in text_list:
    #         text_list.append('{"mask"}')
    #     if '{"placeholder":"text_b"}' not in text_list:
    #         text_list.append('{"placeholder":"text_b"}')
    #     return text_list


class VerbalizerGenerator:
    r"""
    This is the automatic label word search implementation in `LM-BFF <https://arxiv.org/pdf/2012.15723.pdf>`_.

    Args:
        model (:obj:`PretrainedModel`): A pre-trained model for label word generation.
        tokenizer (:obj:`PretrainedTokenizer`): The corresponding tokenize.
        candidate_num (:obj:`Optional[int]`): The number of label word combinations to generate. Validation will then be performed on each combination. Defaults to 100.
        label_word_num_per_class (:obj:`Optional[int]`): The number of candidate label words per class. Defaults to 100.
    """
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 candidate_num: Optional[int] = 100,
                 label_word_num_per_class: Optional[int] = 100):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_num = candidate_num
        self.label_word_num_per_class = label_word_num_per_class
        self.probs_buffer, self.labels_buffer = None, None

    def register_buffer(self, data):
        self.model.eval()
        with torch.no_grad():
            inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
            forward_keys = signature(inner_model.forward).args
            input_batch = {key: data[key] for key in data if key in forward_keys}
            logits = self.model.forward(**input_batch).logits[data['loss_ids']==1]
        logits = F.softmax(logits.detach(),dim=-1)
        if self.probs_buffer is None:
            self.probs_buffer = logits
            self.labels_buffer = data.label.detach()
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, data.label.detach()])
    @abstractmethod
    def post_process(self, word: str):
        r"""
        Post-processing for generated labrl word.

        Args:
            word (:obj:`str`): The original word token.

        Returns:
            processed_word (:obj:`str`): The post-processed token.
        """
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return word.lstrip(' ')
        elif isinstance(inner_model, BertForMaskedLM):
            return word
        else:
            raise RuntimeError("{} is not supported yet".format(type(inner_model))) # TODO add more model


    @abstractmethod
    def invalid_label_word(self, word: str):
        r"""
        Decide whether the generated token is a valid label word. Heuristic strategy can be implemented here, e.g. requiring that a label word must be the start token of a word.

        Args:
            word (:obj:`str`): The token.
        Returns:
            is_invalid (:obj:`bool`): `True` if it cannot be a label word.
        """
        inner_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        if isinstance(inner_model, RobertaForMaskedLM):
            return (not word.startswith(' '))
        elif isinstance(inner_model, BertForMaskedLM):
            return False
        else:
            raise RuntimeError("{} is not supported yet".format(type(inner_model))) # TODO


    def _show_verbalizer(self):
        logger.info("Verbalizer is {}".format(self.label_words))


    def _find_verbalizer(self):
        logger.info("Finding verbalizer ...")
        label_words =  self._get_top_words()
        label_words = self._get_top_group(candidates=label_words)
        return label_words

    def _eval_group(self, group):
        label_logits = self.probs_buffer[:,torch.tensor(group)]
        preds = torch.argmax(label_logits, axis=-1)
        correct = torch.sum(preds == self.labels_buffer)
        return (correct / len(self.labels_buffer)).item()

    def _get_top_group(self, candidates: List[List[int]]):
        groups = list(itertools.product(*candidates))
        group_scores = list(map(self._eval_group, groups))

        # Take top-n.
        best_idx = np.argsort(-np.array(group_scores))[:self.candidate_num]
        best_groups = [groups[i] for i in best_idx]
        return best_groups

    def _get_top_words(self):
        label_words_ids = []
        for label_id in torch.unique(self.labels_buffer):
            scores = self.probs_buffer[self.labels_buffer==label_id].mean(axis=0).cpu().numpy()
            kept = []
            for i in np.argsort(-scores):
                word = self.tokenizer.convert_ids_to_tokens([i])[0]
                if self.invalid_label_word(word):
                    continue
                kept.append(i)
            label_words_ids.append(kept[:self.label_word_num_per_class])
        return label_words_ids

    @classmethod
    def from_config(cls, config: CfgNode, **kwargs,):
        r"""
        Returns:
            verbalizer_generator (:obj:`VerbalizerGenerator`)
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer_generator = cls(**init_dict)
        return verbalizer_generator


    def release_memory(self):
        self.model = self.model.cpu()

    def generate(self):
        r"""
        Generate label words.

        Returns:
            label_words (:obj:`List[List[str]]`): A list of generated label word.
        """

        self.label_words_ids = self._find_verbalizer()
        self.label_words = [[self.post_process(word) for word in self.tokenizer.convert_ids_to_tokens(i)] for i in self.label_words_ids]
        self._show_verbalizer()
        return self.label_words


class RobertaVerbalizerGenerator(VerbalizerGenerator):
    def __init__(self,
                 model: RobertaForMaskedLM,
                 tokenizer: RobertaTokenizer,
                 candidate_num: Optional[int] = 100,
                 label_word_num_per_class: Optional[int] = 100):
        super().__init__(
                        model = model,
                        tokenizer = tokenizer,
                        candidate_num = candidate_num,
                        label_word_num_per_class = label_word_num_per_class)

    def invalid_label_word(self, word: str):
        return (not word.startswith('Ġ'))


    def post_process(self, word: str):
        return word.lstrip('Ġ')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------    
'''