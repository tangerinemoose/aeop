"""
Issues:
    Consider updating prompting technique

General Workflow:
    ##STEP1.1## - Update file location containing Correctness of predictions
    ##STEP1.2## - Prep API Configuration
    ##STEP1.3## - Run a loop to use each category as the validation set (based on cities list)

"""
import openai
from openai import OpenAI
#from dotenv import load_dotenv
import os
import csv

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

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
# =============================================================================
# LOAD DATA - ##STEP1.1##
# =============================================================================

with  open('./post-roberta/outputs/chicago_il.csv', encoding='utf-8')  as inFile :
    trainPrompt = '''Given a "Sentence", predict if the classification output is correct (1) or incorrect (0). Only return either "1" or "0", do not return any other tokens'''
    iCSV = csv.reader(inFile)

    count = 0 #Need to reduce context length for GPT code
    for line in iCSV:
        if count == 500:
          sentence = line[0]
          label = int(line[1])
          trainPrompt = trainPrompt + f"\n\n" + f"Sentence:\n{sentence}\n" + f"Label:\n{label}\n"
        count += 1
#print(trainPrompt)

# =============================================================================
# API CONFIG - ##STEP1.2##
# =============================================================================

# Load environment variables from the .env file
#load_dotenv('./data/.env')

# Get the API key from the environment variable
#api_key = os.getenv("OPENAI_API_KEY")
api_key = ''

# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key
client = OpenAI(api_key=api_key)

# =============================================================================
# PROMPTING - ##STEP1.3##
# =============================================================================

cities = ['baltimore_md','chicago_il','columbus_oh','detroit_mi','el_paso','houston_tx','indianapolis_in','los_angeles','memphis_tn','miami_fl','new_orleans','new_york','philadelphia_pa','phoenix_az','san_antonio']
cities = ['san_antonio']
for c in cities:
  data_dict = {}
  test_list = []
  test_labs = []
  #header = True
  header = False
  #with  open('./data/dev.csv', encoding='utf-8')  as inFile :
  with  open(r'./geo_dataset_fixed.csv', encoding ='utf-8') as inFile:
      iCSV = csv.reader(inFile)

      sentence_list = []
      label_list = []
      for line in iCSV:
          if header:
              header = False
          else:
              if line[0] == c:
                  if line[1] in ('0','1'):
                      sentence = line[2]
                      sentence_list.append(sentence)
                      label = int(line[1])
                      label_list.append(label)

  data_dict['label'] = label_list                    
  data_dict['cqa'] = [f"Sentence:\n{sentence} ####\n\nLabel:\n{label}\n\n" for sentence, label in zip(sentence_list, label_list)]
  true_false_list = []
  for cqa in data_dict['cqa']:
      response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
          {
            "role": "system",
            "content": trainPrompt
          },
          {
            "role": "user",
            "content": cqa
          }
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
      )
      true_false_list.append(response.choices[0].message.content)
      
  #print(true_false_list)
  label_prediction_list = [] 
  for item in true_false_list:
      #if item.upper() == "TRUE":
      if item.upper() == '1':
          label_prediction_list.append(1)
      #if item.upper() == "FALSE":
      if item.upper() == '0':
          label_prediction_list.append(0)
              
  data_dict['prediction'] = label_prediction_list
  #f1_macro = f1_score(data_dict['label'], data_dict['prediction'], average = 'macro')
  #print('Macro F1 Score:', f1_macro)
  
  #preds = np.argmax(predictions.predictions, axis=-1)
  f1 = f1_score(data_dict['label'], data_dict['prediction'])
  acc = accuracy_score(data_dict['label'], data_dict['prediction'])
  #print("CITY:",city)
  #print("F1 Score:", f1)
  #print("Accuracy:", acc)
  try:
      #roc = roc_auc_score(real_labs,preds_score)
      roc = roc_auc_score(data_dict['label'], data_dict['prediction'])
  except:
      roc = 'ROC: Only one class present in y_true. ROC AUC score is not defined in that case.'
  print('RESULTS', c, 1-sum(data_dict['prediction'])/len(data_dict['prediction']), acc, roc)  
