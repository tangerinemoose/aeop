"""
Issues:
    The RoBERTa post-model had issues generating realistic performance values, in amazon_post_roberta_roberta.py. Initial attempts yielded strange results with accuracy, ROC, etc. 

General Workflow:
    ##STEP1.1## - Prep RoBERTa model configuration, including setting up checkpoints
    ##STEP1.2## - Update file location containing Correctness of predictions and extract to lists
    ##STEP1.3## - Train model after tokenize step for training and validation sets, used for RoBERTa
    ##STEP1.4## - Run a loop to use each category as the validation set (based on cities list)

"""

from datasets import Dataset
import pandas as pd
import evaluate
import sys
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback
)
from copy import deepcopy
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from drift_utils import *

import csv

def main(): 

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        print("This epoch's F1:", f1, "Acc:", acc)
        return {
            "f1_score": f1,
        }
    

    ##STEP1.1##
    training_args = TrainingArguments( 
        'test-trainer', # <- HERE
        learning_rate=3e-5,
        warmup_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        #fp16=True, #Enable if using GPU, https://stackoverflow.com/questions/68007097/getting-a-mixed-precison-cuda-error-while-running-a-cell-trying-to-fine-tuning-a
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=1,
        greater_is_better=True,
        metric_for_best_model="f1_score",
    )
    
    checkpoint = "roberta-base"
    #checkpoint = r"./post-roberta/models/test-trainer-satx/checkpoint-600" #####UPDATE

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation = True, padding = True)

    ##STEP1.2##
    X = []
    y = []
    header = False
    with  open(r'./post-roberta/amazonpredictions/All_Beauty.csv', encoding ='utf-8') as inFile: ####UPDATE
        iCSV = csv.reader(inFile)
        for line in iCSV:
            if header:
                header = False
            X.append(line[0])
            y.append(int(line[1]))

    ##STEP1.3##
    cqa_list, dev_list, cqa_labs, dev_labs = train_test_split(X, y, test_size=0.2, random_state=123)

    cqa_Dataset = Dataset.from_dict({'cqa': cqa_list, 'labels':cqa_labs})
    cqa_Dataset = cqa_Dataset.with_format("torch")
    dev_Dataset = Dataset.from_dict({'cqa': dev_list, 'labels':dev_labs})
    dev_Dataset = dev_Dataset.with_format("torch")


    def cqa_tokenize_function(batch):
        return tokenizer(
            batch['cqa'],
            truncation=True,
            padding='max_length',
            max_length=256
        )

    tokenized_train_dataset = cqa_Dataset.map(cqa_tokenize_function)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['cqa'])

    tokenized_dev_dataset = dev_Dataset.map(cqa_tokenize_function)
    tokenized_dev_dataset = tokenized_dev_dataset.remove_columns(['cqa'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # trainer.add_callback(CustomCallback(trainer)) 
    ####ENABLE TO TRAIN
    trainer.train()


    ##STEP1.4##
    cities = ['All_Beauty','Appliances','Baby_Products','Digital_Music','Gift_Cards','Grocery_and_Gourmet_Food','Handmade_Products','Health_and_Personal_Care','Movies_and_TV','Musical_Instruments','Office_Products','Pet_Supplies','Software','Subscription_Boxes','Video_Games']
    cities = ['All_Beauty','Appliances','Baby_Products']
    for c in cities:
        test_list = []
        test_labs = []
        #header = True
        header = False
        with  open(r'./amazondataset.csv', encoding ='utf-8') as inFile:
            iCSV = csv.reader(inFile)
            for line in iCSV:
                if header:
                    header = False
                else:
                    if line[2] == c:
                        dev = line[1]
                        test_list.append(dev)
                        labels = int(line[0])
                        test_labs.append(labels)

    
        D2 = test_list
        test_Dataset = Dataset.from_dict({'cqa': test_list, 'labels':test_labs})
        test_Dataset = test_Dataset.with_format("torch")

        tokenized_test_dataset = test_Dataset.map(cqa_tokenize_function)
        tokenized_test_dataset = tokenized_test_dataset.remove_columns(['cqa'])
        predictions = trainer.predict(tokenized_test_dataset)
        #preds = np.argmax(predictions.predictions, axis=-1)
        #preds_score = predictions.predictions[:,1]
        #accuracy = accuracy_score(test_labs, preds)
        preds = np.argmax(predictions.predictions, axis=-1)
        f1 = f1_score(predictions.label_ids, preds, average="macro")
        acc = accuracy_score(predictions.label_ids, preds)
        #print("CITY:",city)
        print("F1 Score:", f1)
        print("Accuracy:", acc)
        try:
            #roc = roc_auc_score(real_labs,preds_score)
            roc = roc_auc_score(predictions.label_ids, preds)
        except:
            roc = 'ROC: Only one class present in y_true. ROC AUC score is not defined in that case.'
        print('RESULTS', c, preds.sum(), preds.shape[0], 1-preds.sum()/preds.shape[0], acc, roc)    
        sys.stdout.flush()
    sys.stdout.flush()
    
    
if __name__ == '__main__':
    main()