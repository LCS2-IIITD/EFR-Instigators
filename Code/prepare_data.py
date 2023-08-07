
import os
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import re
import copy
import time
import tqdm

os.chdir('../../')

MAX_NO_OF_SPEAKERS = 8
MAX_DIALOGUE_LEN   = 33
original_labels    = ['abuse', 'adoration', 'annoyance', 'awkwardness', 'benefit', 'boredom', 'calmness', 'challenge', 'cheer', 'confusion', 'curiosity', 'desire', 'excitement', 'guilt', 'horror', 'humour', 'impressed', 'loss', 'nervousness', 'nostalgia', 'pain', 'relief', 'satisfaction', 'scold', 'shock', 'sympathy', 'threat']
train_count        = [31, 190, 1051, 880, 220, 78, 752, 214, 534, 486, 545, 180, 867, 216, 280, 153, 257, 351, 398, 65, 36, 173, 136, 94, 372, 209, 263]

EMOTIONS           = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
OUTPUT_MASK        = pickle.load(open('dump_files/mask/top10_mask0_dict.pkl', 'rb'))['mask_tensor']

#@title Import Files { display-mode: "both" }
sent_model = 'roberta-base-nli-stsb-mean-tokens' #@param {type:"string"}

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

from torch import nn, optim


print('tr version', transformers.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device => ",device, ' torch ', torch.__version__)

# for finetuned sentence embeddings
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        op = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = self.drop(op[1])
        return self.out(output), op[1]

# load finetuned roberta model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_finetuned = EmotionClassifier(7).to(device)
roberta_tf_checkpoint = torch.load('dump_files/finetuned/best_model_state_roberta.bin', map_location=torch.device(device))
roberta_finetuned.load_state_dict(roberta_tf_checkpoint)
print('model loaded')


# Helper functions

def readData():
    train_csv = pd.read_excel("./data/MELD-I-partial.xlsx", sheet_name = 'Train')
    dev_csv = pd.read_excel("./data/MELD-I-partial.xlsx", sheet_name = 'Dev')
    test_csv = pd.read_excel("./data/MELD-I-partial.xlsx", sheet_name = 'Test')
    return train_csv, dev_csv, test_csv

def preprocessInstigator(str_ins):
    in1 = str_ins.lower()
    in1 = re.sub('[ ]',',', in1)
    ins = re.sub('[.]',',', in1)
    ins = ins.split(',')
    final = []
    for i in ins:
        if(len(i)>2):
            final.append(i)
    return list(set(final))

def prepareData(data):
    # 1. Remove all empty rows and keep only ['chat_id', 'emotion', 'speaker', 'utterance', 'instigator_flag', 'instigators'] columns
    # 2. One Hot Encode Emotion Labels in the data from step 1- emotion
    # 3. Segregate Instigator labels(Multi-label) So one Hot Encode for each utterance - labels
    # 4. Return dataset from step 2 and onehot encoded instogator labels
    unannotated = []
    data_list   = []
    for i in tqdm.tqdm(range(len(data))):
        # print(data["Chat_Id"][i])
        each_row = []
#         try:
        if type(data["Chat_Id"][i]) == str:
            print('\nContinue: ',data["Chat_Id"][i])
            continue
        if not np.isnan(data["Chat_Id"][i]):
            utt = str(data["Utterance"][i])
            emotion = data["Emotion_name"][i]
            speaker = data["Speaker"][i]
            chat_id = int(data["Chat_Id"][i])
            ann = int(data["Annotate(0/1)"][i])
            labels = data["Label"][i]
            if(ann == 1 and type(labels) != str):
                unannotated.append(chat_id)
            if(type(labels) == str):
                labels = preprocessInstigator(labels)

            each_row = [chat_id, emotion, speaker, utt, ann, labels]
#         except:
#             print('\nException in ',i)
#             display(data.iloc[i:i+1,:])
        if(len(each_row) != 0):
            data_list.append(each_row)

    ## Step 2
    col = 'emotion'
    data_df = pd.DataFrame(data_list, columns = ['chat_id', 'emotion', 'speaker', 'utterance', 'instigator_flag', 'instigators'])
#     data_ohe = pd.concat([data_df, pd.get_dummies(data_df[col], prefix = col)], axis = 1).drop([col], axis = 1)
    data_ohe = pd.concat([data_df, pd.get_dummies(data_df[col], prefix = col)], axis = 1)
   
    leftout_emo = np.zeros(len(data_ohe), dtype = 'int')

    for ei in EMOTIONS:
        temp_emo = 'emotion_'+ei
        if temp_emo not in data_ohe.columns:
            print(temp_emo, 'not present in data')
            data_ohe[temp_emo] = leftout_emo


    ## Step 3 Remove unannotated
    unannotated = list(sorted(set(unannotated)))
    # print(unannotated)
    # print('\n\nshape before:', data_ohe.shape)
    for cid in unannotated:
        # print(cid, type(cid))
        data_ohe = data_ohe.loc[data_ohe['chat_id'] != cid]
    data_ohe = data_ohe.reset_index(drop = True)
    # print('\n\nshape after:', data_ohe.shape)
    

    ## Step 3    
    data_instigators = []
    for i in tqdm.tqdm(range(len(data_ohe))):
        each_row = np.zeros(len(original_labels), dtype = float)
        if(type(data_ohe['instigators'][i]) == list):
            for ins in data_ohe['instigators'][i]:
                if ins in original_labels:
                    idx = original_labels.index(ins)
                    each_row[idx] = 1
                else:
                    print(i, ins, data_ohe['utterance'][i])
        data_instigators.append(each_row)

    ins_labels = ['instigator_'+i for i in original_labels]
    final_data_labels = pd.DataFrame(data_instigators, columns = ins_labels)
    # final_data_labels['chat_id'] = data_ohe['chat_id']
    final_data_labels.insert(0, 'chat_id', data_ohe['chat_id'])

    ## Step 4
    return data_ohe, final_data_labels, unannotated

def sentenceEmbedFineTuned(data, model = roberta_finetuned):
    i = 0
    sentence_embeddings = []
    while i < len(data):
        chat_id = data['chat_id'][i]
        # sent_emb = model.encode('')
        while i < len(data) and data['chat_id'][i] == chat_id:
            utt = data['utterance'][i]
            encodings = roberta_tokenizer.encode_plus(utt, max_length=100, padding = 'max_length', add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True, truncation=True, return_tensors='pt').to(device)
            utt_emb = roberta_finetuned(encodings['input_ids'], encodings['attention_mask'])[1].detach().tolist()[0]
            utt_emb = np.round(utt_emb, decimals = 10)
            # utt_emb = model.encode(utt)
            sent_emb = utt_emb
            i += 1
            sentence_embeddings.append(copy.deepcopy(sent_emb))
    data['sentence_embeddings'] = sentence_embeddings
    df_sent = pd.DataFrame(sentence_embeddings)
    return data, df_sent

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def oheSpeaker(speakers):
    sp_set = unique(speakers)
    # print(sp_set)
    ohe_speakers_padded = np.zeros([MAX_DIALOGUE_LEN, MAX_NO_OF_SPEAKERS], dtype = float)
    ohe_speakers        = np.zeros([len(speakers), MAX_NO_OF_SPEAKERS], dtype = float)
    for i in range(len(speakers)):
        sidx = sp_set.index(speakers[i])
        ohe_speakers[i][sidx]        = 1.0
        ohe_speakers_padded[i][sidx] = 1.0
    return ohe_speakers, ohe_speakers_padded

def padData(data):
    # pad data 
    # ex : sentence embeddings 
    # current shape : (sentence_embeddings.shape = len(dialogue)*768)
    # pad it to MAX_DIALOGUE_LEN*768 by adding ([0]*768)*(MAX_DIALOGUE_LEN - len(dialogue)) 
    
    pad_len     = MAX_DIALOGUE_LEN - len(data)     # 33 - 5(dialogue_len) = 28 for first train dialogue
    pad_emb_len = len(data[0])                     # 768 for roberta (embedding vector size) we use this to create the same sized 0 vector
    pad_emb     = [[0.0]*pad_emb_len]*pad_len                       # [[0, 0, ... pad_emb_len times], [0, ...] ... pad_len times]
    padded_emb = data + pad_emb
    return padded_emb

def toFloat(lst):
    return [float(i) for i in lst]

def groupDialogues(data, labels):
    data_list = []
    i = 0
    while i < len(data):
        chat_id = data['chat_id'][i]
        speaker_list = []
        emotion_str_list = []
        emotion_ohe_list = []
        emotion_flip_list = []
        instigator_flag_list = []
        sent_emb_list = []
        label_list = []
        label_inst_only = []
        # utt_idx = 0
        while i < len(data) and data['chat_id'][i] == chat_id:
            speaker = data['speaker'][i]
            emo_str = data['emotion_str'][i]
            emotion_ohe = [data['emotion_anger'][i], data['emotion_disgust'][i], data['emotion_fear'][i], data['emotion_joy'][i], data['emotion_neutral'][i], data['emotion_sadness'][i], data['emotion_surprise'][i]]
            instigator_flag = data['instigator_flag'][i]
            sent_emb = data['sentence_embeddings'][i]
            label = labels.values[i][1:]

            speaker_list.append(speaker)
            emotion_str_list.append(emo_str)
            emotion_ohe_list.append(toFloat(emotion_ohe))
            instigator_flag_list.append(instigator_flag)
            sent_emb_list.append(sent_emb)
            label_list.append(toFloat(label))
            if(np.sum(label)>0):
                # label_inst_only.append(utt_idx)
                label_inst_only.append(label)
            i += 1
            # utt_idx += 1
#         print(speaker_list, emotion_str_list)
        tg_speaker = speaker_list[-1]
        second_last = -1
        for sp in range(len(speaker_list[:-1])):
            if(speaker_list[sp] == tg_speaker):
                second_last = sp
        emo_flip =  str(emotion_str_list[second_last])+'->'+str(emotion_str_list[-1])
        emotion_flip_list.append(emo_flip)
#         print(emo_flip)

        speaker_ohe_list, padded_speaker_ohe_list = oheSpeaker(speaker_list)
        padded_sent_emb  = padData(sent_emb_list)
        pad_labels       = padData(label_list)
        padded_emotions  = padData(emotion_ohe_list)
        padded_flags     = instigator_flag_list + [0]*(MAX_DIALOGUE_LEN - len(instigator_flag_list))
        target_idx       = len(instigator_flag_list)   
        data_list.append([chat_id, speaker_list, speaker_ohe_list, padded_speaker_ohe_list, emotion_flip_list, emotion_str_list, emotion_ohe_list, padded_emotions, instigator_flag_list, padded_flags, sent_emb_list, padded_sent_emb, target_idx, label_inst_only, label_list, pad_labels])
    
    df_group = pd.DataFrame(data_list, columns = ['chat_id',  'speaker_str', 'speaker_ohe', 'padded_speaker_ohe', 'emotion_flip', 'emotion_str', 'emotions', 'padded_emotions' ,'instigator_flag', 'padded_instigator_flag', 'sentence_embeddings', 'padded_sentence_embeddings', 'target_idx', 'label_instigator_only', 'labels', 'padded_labels'])

    return df_group



def checkprepared():
	try:
		train_path  = 'dump_files/train_files_9emo2.pkl'
		dev_path    = 'dump_files/dev_files_9emo2.pkl'
		test_path   = 'dump_files/test_files_9emo2.pkl'
		x = pickle.load(open(train_path, "rb"))
		train_final   = x[0]
		train_labels  = x[1]
		train_sent    = x[2]
		train_grouped = x[3]

		x = pickle.load(open(dev_path, "rb"))
		dev_final   = x[0]
		dev_labels  = x[1]
		dev_sent    = x[2]
		dev_grouped = x[3]


		x = pickle.load(open(test_path, "rb"))
		test_final   = x[0]
		test_labels  = x[1] 
		test_sent    = x[2]
		test_grouped = x[3]
		print('preprocessed data already present')
		return 1
	except:
		print('Preprocessed data not present')
		return -1



print('\n\n(1/3)Reading and preparing data')
train_df, dev_df, test_df = readData()

start = time.time()

train_ohe, train_labels, tr_unan    =   prepareData(train_df)
dev_ohe, dev_labels, dev_unan       =   prepareData(dev_df)
test_ohe, test_labels, te_unan      =   prepareData(test_df)

print('(1/3)Time taken : ',time.time() - start)


print('\n\n(2/3)Preparing sentence embeddings(pretrained Roberta)')

start = time.time()

train_final, train_sent = sentenceEmbedFineTuned(train_ohe)
print('Train Done')
dev_final, dev_sent     = sentenceEmbedFineTuned(dev_ohe)
print('Dev Done')
test_final, test_sent   = sentenceEmbedFineTuned(test_ohe)
print('Test Done')

print('(2/3)Time taken : ',time.time() - start)

print('\n\n(3/3)Finalizing and grouping data')

train_final['emotion_str'] = train_ohe['emotion']
dev_final['emotion_str'] = dev_ohe['emotion']
test_final['emotion_str'] = test_ohe['emotion']


start = time.time()

train_grouped = groupDialogues(train_final, train_labels)
print('Train Done')
dev_grouped = groupDialogues(dev_final, dev_labels)
print('Dev Done')
test_grouped = groupDialogues(test_final, test_labels)
print('Test Done')

print('(3/3)Time taken : ',time.time() - start)

with open("dump_files/train_files_27.pkl","wb") as f:
    pickle.dump([train_final, train_labels, train_sent, train_grouped],f)
with open("dump_files/dev_files_27.pkl","wb") as f:
    pickle.dump([dev_final, dev_labels, dev_sent, dev_grouped],f)
with open("dump_files/test_files_27.pkl","wb") as f:
    pickle.dump([test_final, test_labels, test_sent, test_grouped],f)
print('preprocessed files saved at ./dump_files')

