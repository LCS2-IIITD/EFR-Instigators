import os
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import re
import copy
import time

# Scoring
from sklearn.metrics import classification_report, f1_score

os.chdir('../../')
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device => ",device, ' torch ', torch.__version__)


# hyper parameters
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
#@title Hyper Parameters { display-mode: "both" }

EPOCHS             = 20
MAX_NO_OF_SPEAKERS = 8
MAX_DIALOGUE_LEN   = 33
original_labels    = ['abuse', 'adoration', 'annoyance', 'awkwardness', 'benefit', 'boredom', 'calmness', 'challenge', 'cheer', 'confusion', 'curiosity', 'desire', 'excitement', 'guilt', 'horror', 'humour', 'impressed', 'loss', 'nervousness', 'nostalgia', 'pain', 'relief', 'satisfaction', 'scold', 'shock', 'sympathy', 'threat']
train_count        = [31, 190, 1051, 880, 220, 78, 752, 214, 534, 486, 545, 180, 867, 216, 280, 153, 257, 351, 398, 65, 36, 173, 136, 94, 372, 209, 263]

EMOTIONS           = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
OUTPUT_MASK        = pickle.load(open('dump_files/mask/top10_mask0_dict.pkl', 'rb'))['mask_tensor']

# DataLoader Hyperparamaters
BATCH_SIZE = 64

# Module 1 hyperparamaters(speaker_specific_emotion_sequence) : GRU n-n
input_size_1  = 7
hidden_size_1 = 10 
num_layers_1  = 2 
output_size_1 = 10


# Module 2 hyperparamaters(utterance_context) : Transformer Enc
input_size_2 = 768
n_head_2     = 4
dm_ff_2      = 2048
dp_2         = 0.2
num_layers_2 = 4 
act_fn_2     = 'relu'

# Module 3 hyperparamaters(speaker_context) : Transformer Enc
input_size_3 = 8
n_head_3     = 4
dm_ff_3      = 2048
dp_3         = 0.2
num_layers_3 = 4 
act_fn_3     = 'relu'

# Module 4 hyperparamaters(global_emotion_sequence) : GRU
input_size_4  = 7
hidden_size_4 = 10 
num_layers_4  = 2 
output_size_4 = 7

# Final Model Hyperparamerters:
fc1_out = 800
fc2_out = 800
fc3_out = 400
fc4_out = 100
fc5_out = len(original_labels)

learning_rate = 0.0001

# Classes

# Loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        # print('BCE Loss\n', BCE_loss)
        # print('Loss\n', F_loss)
        # print('inputs\n', inputs)
        # print('Targets\n', targets)
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# DataLoader

class MELDCollate:
    def __init__(self, pad_value = 0):
        self.pad_value = pad_value
    def __call__(self, batch):
        speaker             = pad_sequence([item[0]['speaker'] for item in batch]).permute(1,0,2)
        emotion             = pad_sequence([item[0]['emotion'] for item in batch]).permute(1,0,2)
        instigator_flag     = pad_sequence([item[0]['instigator_flag'] for item in batch]).permute(1,0)
        sentence_embeddings = pad_sequence([item[0]['sentence_embeddings'] for item in batch]).permute(1,0,2)
        target_idx          = torch.tensor([item[0]['target_idx'] for item in batch])
        # print('\noriginal list : ',[item[0]['speaker'] for item in batch], '\n\npadded list : ', speaker)
        labels              = pad_sequence([item[1]['labels'] for item in batch]).permute(1,0,2)

        dict_x = { 'speaker': speaker, 'emotion':emotion, 'instigator_flag':instigator_flag, 'sentence_embeddings':sentence_embeddings, 'target_idx':target_idx}
        dict_y = {'labels': labels}

        return dict_x, dict_y

class MELDDataset(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))[3]
        self.len = len(self.data)

        print(self.data.columns)
        # chat_id   speaker_str speaker_ohe padded_speaker_ohe  emotions    padded_emotions instigator_flag padded_instigator_flag  sentence_embeddings padded_sentence_embeddings  labels  padded_labels        

    def __getitem__(self, index):
        dict_x = {}
        dict_x['speaker'] = torch.tensor(self.data['speaker_ohe'][index], dtype=torch.float32)
        dict_x['emotion'] = torch.tensor(self.data['emotions'][index], dtype=torch.float32)
        dict_x['instigator_flag'] = torch.tensor(self.data['instigator_flag'][index], dtype=torch.int)
        dict_x['sentence_embeddings'] = torch.tensor(self.data['sentence_embeddings'][index], dtype=torch.float32)
        dict_x['target_idx'] = torch.tensor(self.data['target_idx'][index], dtype=torch.int)
        dict_y = {}
        dict_y['labels'] =  torch.tensor(self.data['labels'][index], dtype=torch.float32)

        return dict_x, dict_y

    def __len__(self):
        return self.len


class MELDDatasetFullPad(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))[3]
        self.len = len(self.data)

        print(self.data.columns)
        # chat_id   speaker_str speaker_ohe padded_speaker_ohe  emotions    padded_emotions instigator_flag padded_instigator_flag  sentence_embeddings padded_sentence_embeddings  labels  padded_labels
        

    def __getitem__(self, index):
        dict_x = {}
        dict_x['speaker'] = torch.tensor(self.data['padded_speaker_ohe'][index], dtype=torch.float32)
        dict_x['emotion'] = torch.tensor(self.data['padded_emotions'][index], dtype=torch.float32)
        dict_x['instigator_flag'] = torch.tensor(self.data['padded_instigator_flag'][index], dtype=torch.int)
        dict_x['sentence_embeddings'] = torch.tensor(self.data['padded_sentence_embeddings'][index], dtype=torch.float32)
        dict_x['target_idx'] = torch.tensor(self.data['target_idx'][index], dtype=torch.int)
        
        dict_y = {}
        dict_y['labels'] =  torch.tensor(self.data['padded_labels'][index], dtype=torch.float32)

        return dict_x, dict_y

    def __len__(self):
        return self.len


class Module1GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Module1GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Since there are maximum of 8 speakers in a dialogue, so we decided to make 8 GRUs one for each speaker.
        self.gru_list= []
        for id in range(MAX_NO_OF_SPEAKERS):
            self.gru_list.append(nn.GRU(input_size, hidden_size, num_layers, batch_first = True))
        self.gru_modules = nn.ModuleList(self.gru_list)
        # self.fc  = nn.Linear(num_layers*hidden_size, output_size)
            
    
    def segregateEmotions(self, emotions, speakers):
        speaker_specific = []
        utt_id = []
        for i in range(MAX_NO_OF_SPEAKERS):
            speaker_tensor = torch.zeros(MAX_NO_OF_SPEAKERS, dtype = float).to(device)
            speaker_tensor[i] = 1
            emo = emotions[torch.nonzero((speakers == speaker_tensor).sum(dim=1) == speakers.size(1))].permute(1,0,2)
            if(emo.size(1) == 0):
                continue
            utt_id.append(torch.nonzero((speakers == speaker_tensor).sum(dim=1) == speakers.size(1)).permute(1,0)[0])
            speaker_specific.append(emo)
#             print('\n emo size : ',emo.size())
#         print('\n emo concat size : ',speaker_specific, utt_id)
        return speaker_specific, utt_id
    
    def applyGRU(self, speaker_specific, utt_id, seq_len):
        speaker_output = torch.zeros(seq_len, self.output_size)  
        for sp_idx in range(len(utt_id)):
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
            out, hn = self.gru_list[sp_idx](speaker_specific[sp_idx], h0)
            for uid in range(utt_id[sp_idx].size(0)):
                speaker_output[utt_id[sp_idx][uid]] = out[0][uid].clone()
        return speaker_output

    def forward(self, x, speakers):
        batch_size = x.size(0)
        seq_len    = x.size(1)
        outputs = []
        for i in range(batch_size):
            speaker_specific, utt_id = self.segregateEmotions(x[i], speakers[i])
            out = self.applyGRU(speaker_specific, utt_id, seq_len)
            outputs.append(out)
        
        final_output = torch.cat([outputs[i].unsqueeze(2) for i in range(len(outputs))], 2).permute(2,0,1)
        
        return final_output
            
        
class Module2TransformerEnc(nn.Module):
    # S, N, E : (seq_len, batch_size, input/embedding_size)
    def __init__(self, input_size, n_head, dim_ff, dp, num_layers, act_fn = 'relu'):
        super(Module2TransformerEnc, self).__init__()
        self.input_size = input_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = n_head, dim_feedforward = dim_ff, dropout=dp, activation=act_fn)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def make_src_mask(self, src): # src_shape : (S, N, E)
        pad_value = torch.zeros(self.input_size).to(device)
        # pad_value shape : (E), value : [0,0,0, ...]
        src = src.transpose(0,1)
        # src_shape : (N, S, E)

        src_mask = torch.all(torch.eq(src,pad_value),2)
        
        # src_mask shape : (N, S), value : for each batch, it is contains seq_len sized tensors and contains true for pad and false for others
        return src_mask

    def forward(self, x):
        # x shape: seq_len, batch_size, input_size 
        # Since batch_first is not a parameter in trasformer so the input must be S, N, E
        x_mask = self.make_src_mask(x)
        out = self.encoder(x, src_key_padding_mask = x_mask)  
        # out shape : (S, N, E)
        return out


class Module3TransformerEnc(nn.Module):
    # S, N, E : (seq_len, batch_size, input/embedding_size)
    def __init__(self, input_size, n_head, dim_ff, dp, num_layers, act_fn = 'relu'):
        super(Module3TransformerEnc, self).__init__()
        self.input_size = input_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = n_head, dim_feedforward = dim_ff, dropout=dp, activation=act_fn)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)


    def make_src_mask(self, src): # src_shape : (S, N, E)
        pad_value = torch.zeros(self.input_size).to(device)
        # pad_value shape : (E), value : [0,0,0, ...]
        src = src.transpose(0,1)
        # src_shape : (N, S, E)

        src_mask = torch.all(torch.eq(src,pad_value),2)
        # print(src_mask)
        # src_mask shape : (N, S), value : for each batch, it is contains seq_len sized tensors and contains true for pad and false for others
        return src_mask

    def forward(self, x):
        # x shape: seq_len, batch_size, input_size 
        # Since batch_first is not a parameter in trasformer so the input must be S, N, E
        x_mask = self.make_src_mask(x)
        out = self.encoder(x, src_key_padding_mask = x_mask)
        # out shape : (S, N, E)
        return out


class Module4GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Module4GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hn = self.gru(x, h0)
        
        # shape of out :  (N, seq_len, hidden_size)     (torch.Size([10, 33, 8])) 
        # shape of hn  :  (num_layers, N, hidden_size)     (torch.Size([2, 10, 8]))
        hn = torch.flatten(hn.permute(1,0,2), start_dim=1)
        # shape of hn  :  (N, num_layers, hidden_size) and then flatten it to (N, num_layers*hiddem_size) 3D to 2D
        output = self.fc(out)
        # shape of output : [N, output_size]

        return output


class FinalModel(nn.Module):
    def __init__(self, 
            input_size_1, hidden_size_1, num_layers_1, output_size_1,      # module 1    
            input_size_2, n_head_2, dm_ff_2, dp_2, num_layers_2, act_fn_2, # module 2
            input_size_3, n_head_3, dm_ff_3, dp_3, num_layers_3, act_fn_3, # module 3
            input_size_4, hidden_size_4, num_layers_4, output_size_4,      # module 4
            fc1_out, fc2_out, fc3_out, fc4_out, fc5_out, dp, masking = False            # final Model parameters
            ):
        super(FinalModel, self).__init__()

        self.masking = masking

        self.module1 = Module1GRU(input_size = input_size_1, num_layers = num_layers_1, hidden_size = hidden_size_1, output_size = output_size_1).to(device)
        self.module2 = Module2TransformerEnc(input_size = input_size_2, n_head = n_head_2, dim_ff = dm_ff_2, dp = dp_2, num_layers = num_layers_2, act_fn = act_fn_2).to(device)
        self.module3 = Module3TransformerEnc(input_size = input_size_3, n_head = n_head_3, dim_ff = dm_ff_3, dp = dp_3, num_layers = num_layers_3, act_fn = act_fn_3).to(device)
        self.module4 = Module4GRU(input_size = input_size_4, num_layers = num_layers_4, hidden_size = hidden_size_4, output_size = output_size_4).to(device)

        
        self.sigmoid = nn.Sigmoid().to(device)
        self.fc1 = nn.Linear(input_size_2+input_size_3, fc1_out).to(device)
        self.classification = nn.Sequential(
                nn.Linear(2*(output_size_1 + fc1_out + output_size_4), fc2_out).to(device),
                nn.ReLU(),
                nn.Dropout(dp), 
                nn.Linear(fc2_out, fc3_out).to(device),
                nn.ReLU(),
                nn.Dropout(dp),
                nn.Linear(fc3_out, fc4_out).to(device),
                nn.ReLU(),
                nn.Dropout(dp),
                nn.Linear(fc4_out, fc5_out).to(device),
                # nn.Sigmoid()
        )
        

    def concatTarget(self, out1234, target_idx):
        batch_size = out1234.size(0)
        
        cat_output = []

        for i in range(batch_size):
            seq_len = out1234.size(1)
            target_tensor = out1234[i][target_idx[i]-1]
            target_tensor = torch.vstack([target_tensor for _ in range(seq_len)])
            concat_tensor = torch.cat((out1234[i], target_tensor), 1)
            concat_tensor = concat_tensor.unsqueeze(0)
            cat_output.append(concat_tensor)
        cat_output = torch.cat(cat_output, 0)
        return cat_output

    def forward(self, x):
        # extract values from dict x
        speaker = x['speaker'].to(device)               # shape (N, 33, 8)
        utt_seq = x['sentence_embeddings'].to(device)   # shape (N, 33, 768)
        emotions = x['emotion'].to(device)              # shape (N, 33, 7)
        instigator_flag = x['instigator_flag'].to(device)
        target_idx      = x['target_idx'].to(device)
        # print('target',target_idx)

        # Generate outputs from each of the modules (1,2,3,4)
        out1 = self.module1(emotions, speaker).to(device)                   # shape (N, 33, output_size_1)
        # print('1 done',out1.size())
        out2 = self.module2(utt_seq.permute(1,0,2)).to(device)              # shape (33, N, 768)
        out2 = out2.permute(1,0,2)                                          # shape (N, 33, 768)
        # print('2 done',out2.size())
        out3 = self.module3(speaker.permute(1,0,2)).to(device)              # shape (33, N, 8)
        out3 = out3.permute(1,0,2)                                          # shape (N, 33, 8)
        # print('3 done', out3.size())
        out4 = self.module4(emotions).to(device)                            # shape (N, 33, output_size_4)
        # print('4 done', out4.size())
        
        # Concat the outputs and add linear layers acc to the model architecture
        out14 = torch.cat((out1, out4), 2).to(device)   # shape (N, 33, output_size_1+output_size_4)
        out23 = torch.cat((out2, out3), 2).to(device)   # shape (N, 33, 768(output_size_1+output_size_4))
        out23 = F.relu(self.fc1(out23))                 # shape (N, 33, fc1_out)

        out1234 = torch.cat((out23, out14), 2).to(device)
        # print('1234 done', out1234.size())
        out1234 = self.concatTarget(out1234, target_idx)  # shape (N, 33, output_size_1 + fc1_out + output_size_4) [N, 33, ~1614]
        # print('1234 done', out1234.size())
        
        # classify
        # out1234 = self.classify(out1234, instigator_flag)
        outputs = []
        indx  = []
    
        for batch_idx in range(out1234.size(0)): # batch_size
            b_out = []
            if(self.masking):
                target_emotion = emotions[batch_idx][target_idx[batch_idx].item()-1].to(device)
                target_emotion = torch.masked_select(torch.arange(7).to(device), torch.all(torch.eq(torch.eye(7).to(device), target_emotion), 1)).item()
                out_mask = OUTPUT_MASK[target_emotion].to(device)
            # print('Emotion : ', EMOTIONS[target_emotion])
            for in_idx in range(out1234.size(1)): # seq len
                if(instigator_flag[batch_idx][in_idx] == 1):
                    op = self.classification(out1234[batch_idx][in_idx]).to(device)
                    # print('Before Mask op and shape',op, op.shape)
                    if(self.masking):
                        op = op.masked_fill(out_mask == 0, -1e9)
                    op = self.sigmoid(op)
                    # print('After Mask op and shape',op, op.shape)
                    outputs.append(op.unsqueeze(0))
                    indx.append((batch_idx, in_idx))

        outputs = torch.cat(outputs, 0)
        
        return outputs, indx  # No of predictions * 27 ( No. of predictions can be more than the batch size), (No of predictions, 2) : batchno, utt_index



# Helper Functions

def valLoss(model, val_dataloader):
    model.eval()
    loss_list = []
    # loop = tqdm.tqdm(enumerate(val_dataloader), total = len(val_dataloader), position=0, leave=True)
    for idx, (x, y) in enumerate(val_dataloader):
        outputs, pred_index = model(x) # list of list of 27 sized tensors (shape: N, instigator flag in each dailogue, 27)
        labels = y['labels'].to(device)
        targets  = torch.ones(outputs.size()).to(device)
        count = 0
        for b,u in pred_index:
            targets[count] = labels[b][u]
            count += 1

        accuracy = 0
        loss = criterion(outputs, targets).to(device)
        loss_list.append(loss.item())
    loss_avg = sum(loss_list)/len(loss_list)
        
    model.train()
    return loss_avg

def predictSamples(model, loader):
    
    model.eval()
    
    dict_df = {}
    loop = tqdm.tqdm(enumerate(loader), total = len(loader), position=0, leave=True)
    all_op = []
    all_truth = []
    with torch.no_grad():
        for idx, (x, y) in loop:
            labels = y['labels']
            out, indices = model(x)
            
            truth = torch.ones(out.size()).to(device)
            out = out.detach()
            
            count = 0
            for b, u in indices:
                truth[count] = labels[b][u]
                all_op.append(out[count].tolist())
                all_truth.append(truth[count].tolist())
                count+=1
    
    for ins in range(len(original_labels)):
        truli = []
        opli  = []
        for i in range(len(all_op)):
            opli.append(all_op[i][ins])
            truli.append(all_truth[i][ins])
        df = pd.DataFrame([])
        df['pred'] = opli
        df['truth'] = truli
        dict_df[original_labels[ins]] = df
    
    model.train()
    
    return dict_df


def train(model, train_loader, dev_loader, criterion, optimizer, scheduler, path,  EPOCHS=20, save_flag = True):
    print('saving set to :', save_flag)
    best_model=''
    min_loss = 1e9
    dev_loss_list = []
    train_loss_list = []

    for epoch in range(EPOCHS):
        loop = tqdm.tqdm(enumerate(train_loader), total = len(train_loader), position=0, leave=True)
        loss_list = []
        loss_avg = 0.0
        for idx, (x, y) in loop:
            # try:
                outputs, pred_index = model(x) # list of list of 27 sized tensors (shape: N, instigator flag in each dailogue, 27)
                labels = y['labels'].to(device)
                targets  = torch.ones(outputs.size()).to(device)

                count = 0
                for b,u in pred_index:
                    targets[count] = labels[b][u]
                    count += 1

                accuracy = 0
                loss = criterion(outputs, targets).to(device)
                loss_list.append(loss.item())
                loss_avg = sum(loss_list)/len(loss_list)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()


                loop.set_description(f'Epoch[{epoch+1}/{EPOCHS}]')
                loop.set_postfix(loss = loss_avg, accuracy = accuracy)

        # scheduler step
        scheduler.step()

        dev_loss = valLoss(model, dev_loader)
        dev_loss_list.append(dev_loss)
        train_loss_list.append(loss_avg)
        print('\tValidation Loss : ', dev_loss)
            # except:
            #     print('\nNo instigators in :', idx)
        if(min_loss >= dev_loss_list[-1] or epoch == EPOCHS-1):
            min_loss = dev_loss_list[-1]
            model_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'batch_size': BATCH_SIZE
                    }
            model_path = path+str(epoch+1)+'_loss'+str(dev_loss_list[-1])+'.pt'
            if epoch != EPOCHS-1:
                best_model = path+str(epoch+1)+'_loss'+str(dev_loss_list[-1])+'.pt'
            print('Best model till now -> ',best_model)
            if save_flag:
                if epoch == EPOCHS-1:
                    model_path = path+'last_model'+str(epoch+1)+'_loss'+str(dev_loss_list[-1])+'.pt'
                    torch.save(model_checkpoint, model_path)
                else:
                    torch.save(model_checkpoint, model_path)

    # plt.plot(np.arange(len(train_loss_list)), train_loss_list, label = 'train', marker = 'o')
    # plt.plot(np.arange(len(dev_loss_list)), dev_loss_list, label = 'val',  marker = 'x')
    # plt.legend()
    # plt.show()
    # print('\nTrain loss', train_loss_list)
    # print('\nVal loss', dev_loss_list)
    return model, best_model

def plot27(dict_df):
    fig, ax = plt.subplots(6,6, figsize=(100,100))
    ci = 0
    cj = 0
    for ci in range(6):
        for cj in range(6):
            if 6*ci+cj>26:
                ax[ci][cj].plot()
                continue
            inst = original_labels[6*ci+cj]
            df_ones = dict_df[inst].loc[dict_df[inst]['truth'] == 1.0]['pred'].to_frame('pred')
            df_zeros = dict_df[inst].loc[dict_df[inst]['truth'] == 0.0]['pred'].to_frame('pred')
            y1 = df_zeros['pred'].values
            x1 = np.arange(0,0+len(y1), 1)
            ax[ci][cj].scatter(x1, y1, label = 'gt: 0')
            y2 = df_ones['pred'].values
            x2 = np.arange(len(x1),len(x1)+len(y2), 1)
            ax[ci][cj].scatter(x2, y2, label = 'gt: 1')
            ax[ci][cj].title.set_text('Instigator : '+inst)
    
    


## implemetation

#load data
train_path  = 'dump_files/train_files_27.pkl'
dev_path    = 'dump_files/dev_files_27.pkl'
test_path   = 'dump_files/test_files_27.pkl'
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


torch.manual_seed(SEED)

train_dataset = MELDDataset(train_path)
dev_dataset   = MELDDataset(dev_path)
test_dataset  = MELDDataset(test_path)

train_loader  = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn= MELDCollate())
dev_loader    = DataLoader(dataset = dev_dataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn= MELDCollate())
test_loader   = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn= MELDCollate())

cc = 2
print('Dataloader With padding (SEQ LEN: Maxlen of batch)')
for epoch in range(EPOCHS):
    print(f'epoch :{epoch+1}/{EPOCHS}')
    for idx, (x, y) in enumerate(train_loader):
        print('batch :',idx+1)
        for i in x.keys():
            print(i, x[i].shape)
        print('labels',y['labels'].shape)
        # print(x['sentence_embeddings'])
        cc -= 1
        if cc == 0:
            break
        print('\n')
    break





## Train Models


# config 1 TGIF
# with mask

TGIF_model  = FinalModel(
            input_size_1 = input_size_1, num_layers_1 = num_layers_1, hidden_size_1 = hidden_size_1, output_size_1 = output_size_1,      # module 1   
            input_size_2 = input_size_2, n_head_2 = n_head_2, dm_ff_2 = dm_ff_2, dp_2 = dp_2, num_layers_2 = num_layers_2, act_fn_2 = act_fn_2, # module 2
            input_size_3 = input_size_3, n_head_3 = n_head_3, dm_ff_3 = dm_ff_3, dp_3 = dp_3, num_layers_3 = num_layers_3, act_fn_3 = act_fn_3, # module 3
            input_size_4 = input_size_4, num_layers_4 = num_layers_4, hidden_size_4 = hidden_size_4, output_size_4 = output_size_4,      # module 4
            fc1_out = fc1_out, fc2_out = fc2_out, fc3_out = fc3_out, fc4_out = fc4_out, fc5_out = fc5_out, dp = 0.2, 
            masking = True                         # final Model parameters
            ).to(device)
# temp_fm


learning_rate = 0.001
criterion = FocalLoss().to(device)
optimizer = optim.Adam(TGIF_model.parameters(), lr=learning_rate)
tr_steps  = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps = tr_steps)

TGIF_model1, TGIF_best_model1 = train(model = TGIF_model, 
                            train_loader = train_loader, dev_loader = dev_loader, 
                            criterion = criterion, optimizer = optimizer, scheduler = scheduler, 
                            path = 'models/27/config1.1_abl_m1234_ep' ,EPOCHS=EPOCHS, save_flag = True)



# config 2 TGIF
# with no mask
TGIF_model2  = FinalModel(
            input_size_1 = input_size_1, num_layers_1 = num_layers_1, hidden_size_1 = hidden_size_1, output_size_1 = output_size_1,      # module 1   
            input_size_2 = input_size_2, n_head_2 = n_head_2, dm_ff_2 = dm_ff_2, dp_2 = dp_2, num_layers_2 = num_layers_2, act_fn_2 = act_fn_2, # module 2
            input_size_3 = input_size_3, n_head_3 = n_head_3, dm_ff_3 = dm_ff_3, dp_3 = dp_3, num_layers_3 = num_layers_3, act_fn_3 = act_fn_3, # module 3
            input_size_4 = input_size_4, num_layers_4 = num_layers_4, hidden_size_4 = hidden_size_4, output_size_4 = output_size_4,      # module 4
            fc1_out = fc1_out, fc2_out = fc2_out, fc3_out = fc3_out, fc4_out = fc4_out, fc5_out = fc5_out, dp = 0.2, 
            masking = False                         # final Model parameters
            ).to(device)


criterion = FocalLoss().to(device)
optimizer = optim.AdamW(TGIF_model.parameters(), lr=0.0001)
tr_steps  = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps = tr_steps)

TGIF_model2, TGIF_best_model3 = train(model = TGIF_model, 
                            train_loader = train_loader, dev_loader = dev_loader, 
                            criterion = criterion, optimizer = optimizer, scheduler = scheduler, 
                            path = 'models/27/config1.2_abl_m1234_ep' ,EPOCHS=EPOCHS, save_flag = True)