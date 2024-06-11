"""
Dataset was manually put together (no code). Various categories were downloaded, and the first 1,000 records from each category were combined to create a combined dataset:
    Original dataset: https://amazon-reviews-2023.github.io/
    columns: rating, review, category
    dataset location: './amazondataset.csv'

Issues:
    - Post-model
        The RoBERTa post-model had issues generating realistic performance values, in amazon_post_roberta_roberta.py. Initial attempts yielded strange results with accuracy, ROC, etc. 
        GPT-code.py contains code for the GPT-3.5 and GPT-4 models. Though this was manageable (only for 3.5) for the geographic dataset, this may need to be revisited with a more appropriate prompting technique.
            GPT has not been attempted yet for the Amazon review dataset.
    - Minor comments
        Consider renaming variables, lists, etc. appropriately (lots of code pulled from other existing code)
        Consider removing cities list and making it work with checkpoints list instead (STEP2.1)
        Consider other methods of switching post-model methods (switch method parameters, separate code, etc)
        

General Workflow:
    1. Run a RoBERTa model
        ##STEP1.1## - Set up checkpoints. For each existing model/category, add a new elif statement and include in checkpoints list. NOTE: For each value in checkpoints list, entire code will run!
        ##STEP1.2## - Extract data from dataset for training (cqa_list and cqa_labs).
        ##STEP1.3## - Extract data from dataset for validation (dev_list and dev_labs). Magazine_Subscriptions is selected as the category of choice for validation.
        ##STEP1.4## - Tokenize step for training and validation sets, used for RoBERTa
        ##STEP1.5## - (OPTIONAL) Enable to train new model instead of utilize existing model. This is only enabled if checkpoint is "roberta-base" (STEP1.1). Update STEP1.2 to ensure appropriate training values are used.
        ##STEP1.6## - Evaluate performance of models
    2. Post-model
        ##STEP2.1## - Generate 'Correctness' by comparing predictions to ground truth.
        ##STEP2.2## - Write Reviews and Correctness to a separate file. Update file location as needed. This can be used for additional post-model testing.
        ##STEP2.3## - Set cities list to determine which categories to do post-modeling on
        ##STEP2.4## - (PATH 1) Build vocabulary and structural drift model
        ##STEP2.5## - (PATH 2) Build TFIDF model
        ##STEP2.6## - Depending on path, perform logistic regression on model
        ##STEP2.7## - Run a loop to use each category as the validation set (based on cities list)
        ##STEP2.8## - Depending on path, perform logistic regression on model
        ##STEP2.9## - Evaluate performance of models
        ##STEP2.10## - Export performance results to a file.
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
    
    ##STEP1.1##
    #checkpoint = "roberta-base"
    checkpoints = ['All_Beauty','Appliances','Baby_Products','Digital_Music','Gift_Cards','Grocery_and_Gourmet_Food','Handmade_Products','Health_and_Personal_Care','Movies_and_TV','Musical_Instruments','Office_Products','Pet_Supplies','Software','Subscription_Boxes','Video_Games']
    performData = [] #Saving performance numbers to file

    for cp in checkpoints:
        performRow = 'RESULTS' + cp
        performData.append(performRow)
        if cp == 'All_Beauty':
            checkpoint = r"./post-roberta/amazonpredictions/models/all_beauty/checkpoint-378"
        elif cp == 'Appliances':
            checkpoint = r"./post-roberta/amazonpredictions/models/appliances/checkpoint-630"
        elif cp == 'Baby_Products':   
            checkpoint = r"./post-roberta/amazonpredictions/models/baby_products/checkpoint-819"
        elif cp == 'Digital_Music':
            checkpoint = r"./post-roberta/amazonpredictions/models/digital_music/checkpoint-1071"
        elif cp == 'Gift_Cards':
            checkpoint = r"./post-roberta/amazonpredictions/models/gift_cards/checkpoint-693"
        elif cp == 'Grocery_and_Gourmet_Food':
            checkpoint = r"./post-roberta/amazonpredictions/models/grocery_gourmet_food/checkpoint-819"
        elif cp == 'Handmade_Products':
            checkpoint = r"./post-roberta/amazonpredictions/models/handmade_products/checkpoint-693"
        elif cp == 'Health_and_Personal_Care':
            checkpoint = r"./post-roberta/amazonpredictions/models/health_and_personal_care/checkpoint-819"
        elif cp == 'Movies_and_TV':
            checkpoint = r"./post-roberta/amazonpredictions/models/movies_and_tv/checkpoint-945"
        elif cp == 'Musical_Instruments':
            checkpoint = r"./post-roberta/amazonpredictions/models/musical_instruments/checkpoint-1197"
        elif cp  == 'Office_Products':
            checkpoint = r"./post-roberta/amazonpredictions/models/office_products/checkpoint-882"
        elif cp == 'Pet_Supplies':
            checkpoint = r"./post-roberta/amazonpredictions/models/pet_supplies/checkpoint-504"
        elif cp == 'Software':
            checkpoint = r"./post-roberta/amazonpredictions/models/software/checkpoint-378"
        elif cp == 'Subscription_Boxes':
            checkpoint = r"./post-roberta/amazonpredictions/models/subscription_boxes/checkpoint-882"
        elif cp == 'Video_Games':
            checkpoint = r"./post-roberta/amazonpredictions/models/video_games/checkpoint-693"
        else:
            checkpoint = "roberta-base"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation = True, padding = True)

        cqa_list = []
        cqa_labs = []

        ##STEP1.2##
        header = False
        with  open(r'./amazondataset.csv', encoding ='utf-8') as inFile:
            iCSV = csv.reader(inFile)
            for line in iCSV:
                if header:
                    header = False
                else:
                    #if line[2] == '': #Enable and update with value if creating new model
                    if line[2] == cp:
                        cqa = line[1]
                        cqa_list.append(cqa)
                        labels = int(line[0])
                        cqa_labs.append(labels)
        #print(cqa_labs)
        cqa_Dataset = Dataset.from_dict({'cqa': cqa_list, 'labels':cqa_labs})
        cqa_Dataset = cqa_Dataset.with_format("torch")

        ##STEP1.3##
        dev_list = []
        dev_labs = []
        header = False
        with  open(r'./amazondataset.csv', encoding ='utf-8') as inFile:
            iCSV = csv.reader(inFile)
            for line in iCSV:
                if header:
                    header = False
                else:
                    if line[2] == 'Magazine_Subscriptions':
                        dev = line[1]
                        dev_list.append(dev)
                        labels = int(line[0])
                        dev_labs.append(labels)
        #print(dev_labs)                
        dev_Dataset = Dataset.from_dict({'cqa': dev_list, 'labels':dev_labs})
        dev_Dataset = dev_Dataset.with_format("torch")

        ##STEP1.4##
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
        

        ##STEP1.5##
        #trainer.add_callback(CustomCallback(trainer))
        if checkpoint == "roberta-base":
            trainer.train()

        ##STEP1.6##
        predictions = trainer.predict(tokenized_dev_dataset)
        print(predictions.predictions.shape, predictions.label_ids.shape)
        preds = np.argmax(predictions.predictions, axis=-1)
        #f1 = f1_score(predictions.label_ids, preds)
        f1 = f1_score(predictions.label_ids, preds, average="macro")
        acc = accuracy_score(predictions.label_ids, preds)
        print("F1 Score:", f1)
        print("Accuracy:", acc)
        #print('predictions',predictions)
        #print('preds',preds)
        

        ##STEP2.1##
        originalPred = pd.Series(preds, index=dev_list, name = 'originalPred')
        originalGround = pd.Series(dev_labs, index=dev_list, name = 'originalGround')
        predGround = pd.concat([originalPred, originalGround], axis=1)
        predGround['Correctness'] = (predGround['originalPred'] != predGround['originalGround']).astype(int)
        tmp = (predictions.label_ids != preds).astype(int)

        #Create D1 and D2
        D1 = cqa_list
        D2 = dev_list

        ##STEP2.2##
        #data = list(zip(D2, tmp))
        #with open('./post-roberta/amazonpredictions/Video_Games.csv', 'w', newline='') as file: #####UPDATE FILE NAME
        #    writer = csv.writer(file)
        #    for row in data:
        #        writer.writerow(row)

        ##STEP2.3##
        cities = ['All_Beauty','Appliances','Baby_Products','Digital_Music','Gift_Cards','Grocery_and_Gourmet_Food','Handmade_Products','Health_and_Personal_Care','Movies_and_TV','Musical_Instruments','Office_Products','Pet_Supplies','Software','Subscription_Boxes','Video_Games']
        
        
        ##STEP2.4##
        #Train on D1
        vocabModel = vocabDrift()
        vocabModel.train(D1)

        D2SentenceProbDict = []
        for sentence in D2:
            D2SentenceProbDict.append(vocabModel.score(sentence))

        #Create vocab probabilities pd series
        vocabProb = pd.Series(D2SentenceProbDict, name='vocab')
        
        #Train on D1
        structureModel = structureDrift()
        structureModel.train(D1)

        D2StructureProbDict = []
        for sentence in D2:
            #print(structureModel.score(sentence))
            D2StructureProbDict.append(structureModel.score(sentence))

        #Create structure probabilities pd series
        structureProb = pd.Series(D2StructureProbDict, name='structure')
        dev_step_2 = pd.concat([vocabProb, structureProb], axis=1)
        
        ##STEP2.5##
        #Enable for TFDIF
        #vec = TfidfVectorizer(ngram_range=(1,1), max_features=400)
        #vec = TfidfVectorizer(ngram_range=(1,1))
        #vec = TfidfVectorizer(ngram_range=(2,2))
        #vec = TfidfVectorizer(ngram_range=(1,2))

        ##STEP2.6##
        logreg = LinearSVC(class_weight='balanced')

        X = dev_step_2 #Enable for LinearSVC
        #X = vec.fit_transform(D2) #Enable for TFDIF
        y = tmp
        logreg.fit(X,tmp)
        
        ##STEP2.7##
        for c in cities:
            test_list = []
            test_labs = []
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
            preds = np.argmax(predictions.predictions, axis=-1)
            preds_score = predictions.predictions[:,1]
            accuracy = accuracy_score(test_labs, preds)

            #Train on D1
            #vocabModel = vocabDrift()
            #vocabModel.train(D1)

            D2SentenceProbDict = []
            for sentence in D2:
                D2SentenceProbDict.append(vocabModel.score(sentence))

            ###
            #Create vocab probabilities pd series
            vocabProb = pd.Series(D2SentenceProbDict, name='vocab')

            #Train on D1
            #structureModel = structureDrift()
            #structureModel.train(D1)

            D2StructureProbDict = []
            for sentence in D2:
                #print(structureModel.score(sentence))
                D2StructureProbDict.append(structureModel.score(sentence))

            structureProb = pd.Series(D2StructureProbDict, name='structure')

                
            real_labs = np.array(test_labs) != preds
            test_step_2 = pd.concat([vocabProb, structureProb], axis=1)

            ##STEP2.8##
            #tmp_X = vec.transform(D2) #Enable for TFDIF
            #reds = logreg.predict(tmp_X)#test_step_2) #Enable for TFDIF
            preds = logreg.predict(test_step_2) #Enable for LinearSVC
            
            
            ##STEP2.9##
            print(preds_score.shape)
            preds_score = logreg.decision_function(test_step_2)
            #preds_score = logreg.decision_function(tmp_X) #Enable for TFDIF
            #roc = roc_auc_score(real_labs,preds_score)
            try:
                roc = roc_auc_score(real_labs,preds_score)
            except:
                roc = 'ROC: Only one class present in y_true. ROC AUC score is not defined in that case.'
            print('RESULTS', c, preds.sum(), preds.shape[0], 1-preds.sum()/preds.shape[0], accuracy, roc)
            performRow = 'RESULTS' + ' ' + str(c) + ' ' + str(preds.sum()) + ' ' + str(preds.shape[0]) + ' ' + str(1-preds.sum()/preds.shape[0]) + ' ' + str(accuracy) + ' ' + str(roc)
            performData.append(performRow)
            #print(preds)

            
            #sys.stdout.flush()
        #Write performance values to file
    
    ##STEP2.10##
    with open('./test.csv', 'w', newline='') as file: #####UPDATE FILE NAME
        writer = csv.writer(file)
        for row in performData:
            print(row)
            writer.writerow([row])
    
if __name__ == '__main__':
    main()