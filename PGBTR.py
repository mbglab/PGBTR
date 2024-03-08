#!user/bin/env python3

# Make a comprehensive evaluation of the PGBTR model.


import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from net_module import *

        
class MyDataset(Dataset):
    """
    Pytorch loads data.
    """
    def __init__(self, data1=None, data2=None, labels=None, use_dis=True):
        self.use_dis = use_dis
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

        if use_dis==True and (data1 is None or data2 is None):
            raise ValueError("Both data1 and data2 are required for dual input.")

    def __len__(self):
        return len(self.labels)        

    def __getitem__(self, index):
        if self.use_dis:
            sample1 = self.data1[index]
            sample2 = self.data2[index]
            label = self.labels[index]
            return sample1, sample2, label
        else:
            sample = self.data1[index]
            label = self.labels[index]
            return sample, label


def load_matrices(indel_list, data_path, use_dis=True):
    """
    Load PDGD matrices.
    """
    xxdata_list = []    # PDGD matrices
    yydata = []         # data label
    zzdata = []         # sample information
    gd_list = []        # genomic distance information
    count_set = [0]
    count_setx = 0

    for i in indel_list:
        xdata = np.load(data_path+'/xdata' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata' + str(i) + '.npy')
        zdata = np.load(data_path+'/zdata' + str(i) + '.npy')
        
        for k in range(len(ydata)):
            xxdata_list.append(xdata[k,:,:,:])
            yydata.append(ydata[k])
            zzdata.append(zdata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        # distance information
        if use_dis==True:
            for k in range(len(zdata)):
                TF_id, TG_id,_ = zdata[k].split('\t')
                if allgenes_info.loc[TG_id ,'Direction'] == '+':
                    TG_pos = allgenes_info.loc[TG_id,'start']
                if allgenes_info.loc[TG_id ,'Direction'] == '-':
                    TG_pos = allgenes_info.loc[TG_id,'end']
                TF_pos = 0
                for id in TF_id.split(','):
                    TF_pos += (allgenes_info.loc[id,'start'] + allgenes_info.loc[id,'end'])/2

                TF_pos = TF_pos/len(TF_id.split(','))
                genomic_distance = abs(TF_pos - TG_pos)
                # Assume that the genome is connected end to end
                if genomic_distance > whole_genome/2:
                    genomic_distance = whole_genome - genomic_distance
                gd_list.append(np.log(genomic_distance + 1))
    
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    
    if use_dis==True:
        return np.array(xxdata_list), yydata_x, np.array(zzdata), np.array(gd_list)
    else:
        return np.array(xxdata_list), yydata_x, np.array(zzdata)

use_dis = True
allgenes_info = pd.read_csv('data/test_data/allgenes_info.csv', index_col=0)    
whole_genome = 4641652  # E. coli
# parameters of CNNBTR
batch_size = 64
learning_rate = 0.01
step_size = 10          # control learning rate changes
epoch_num = 60

data_path = 'data/test_data/PDGD/'


x1_train, y_train, z_train, x2_train = load_matrices(['train'], data_path, use_dis)
x1_test, y_test, z_test, x2_test = load_matrices(['test'], data_path, use_dis)
     
x1_train = np.transpose(x1_train, (0,3,1,2))
x1_train = torch.tensor(x1_train, dtype=torch.float32)
x2_train = torch.tensor(x2_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Define pytorch dataset
dataset = MyDataset(x1_train, x2_train, y_train, use_dis)

val_size = int(0.2 * len(dataset)) # train_data:validation_data = 4:1
train_size = len(dataset) - val_size

# Split dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

x1_test = np.transpose(x1_test, (0, 3, 1, 2))
x1_test = torch.tensor(x1_test, dtype=torch.float32)
x2_test = torch.tensor(x2_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_dataset = MyDataset(x1_test, x2_test, y_test, use_dis)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)            
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define some variables to track the best accuracy and corresponding model weights
best_auc = 0.0
best_model_weights = None
patience = 15  # maximum number of epochs without change
no_improvement_count = 0  # the number of epochs without change during training

model = CNNBTR()
model = model.to(device)

get_loss = nn.BCELoss()  # two classification         
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.98, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

for epoch in range(epoch_num):
    train_running_loss = 0
    train_total_samples = 0
    train_num_correct = 0
    
    model.train()

    loop1 = tqdm(train_loader, desc='Train', total=len(train_loader))
    for inputs1, inputs2, labels in loop1:
        
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)            

        optimizer.zero_grad()  # gradient initialization
        outputs = model(inputs1, inputs2.unsqueeze(1)) 
        loss = get_loss(outputs, labels.unsqueeze(1))  
        loss.backward()  # backpropagation
        optimizer.step()  # update all parameters

        train_running_loss += loss.item() * labels.size(0)               
        train_total_samples += labels.size(0)
        train_y_ = (outputs >= 0.5).float()
        train_num_correct += (train_y_ == labels.unsqueeze(1)).sum()

        train_acc = float(train_num_correct) / train_total_samples
        train_loss = train_running_loss / train_total_samples
        # Updata information
        loop1.set_description(f'Epoch [{epoch + 1}/{epoch_num}]')
        
    # After each epoch, evaluate model on the validation set
    model.eval()  
    val_running_loss = 0
    val_total_samples = 0
    val_num_correct = 0
    val_probs = []
    val_labels = []

    loop2 = tqdm(val_loader, desc='Val', total=len(val_loader))
    with torch.no_grad():
        for inputs1, inputs2, labels in loop2:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            
            val_predictions = model(inputs1, inputs2.unsqueeze(1))
            val_y_ = (val_predictions >= 0.5).float()
            val_num_correct += (val_y_ == labels.unsqueeze(1)).sum()
            val_total_samples += labels.size(0)
            loss = get_loss(val_predictions, labels.unsqueeze(1))
            val_running_loss += loss.item() * labels.size(0)

            val_acc = float(val_num_correct) / val_total_samples                    
            val_loss = val_running_loss / val_total_samples
            val_probs.extend(val_predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_auc = roc_auc_score(val_labels, val_probs)
            loop2.set_postfix(val_auc=val_auc)

    # If the current validation auc_roc exceeds the optimal accuracy, 
    # update the optimal model weights and save the checkpoint
    if val_auc > best_auc:
        best_accuracy = val_auc
        best_model_weights = model.state_dict()
        no_improvement_count = 0  # reset no change count
        checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_accuracy
                }
        torch.save(checkpoint, f'result/test_checkpoint.pth')
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        print(f'No improvement in validation auroc for {patience} epochs. Training stopped.')
        break

    scheduler.step()  # update the learning rate based on iterative epochs

print('Training finished!')

model = CNNBTR()
checkpoint = torch.load(f'result/test_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
# Initialize test set results
test_acc = 0
test_auc = 0
test_num_correct = 0
test_total_samples = 0
test_probs = []
test_labels = []
pred_labels = []
with torch.no_grad():
    for test_batch_data1, test_batch_data2, test_batch_labels in test_loader:
        
        test_batch_data1, test_batch_data2, test_batch_labels = test_batch_data1.to(device), test_batch_data2.to(device), test_batch_labels.to(device)
        test_predictions = model(test_batch_data1, test_batch_data2.unsqueeze(1))
    
        test_y_ = (test_predictions >= 0.5).float()
        test_num_correct += (test_y_ == test_batch_labels.unsqueeze(1)).sum()
        test_total_samples += test_batch_labels.size(0)

        test_probs.extend(test_predictions.cpu().numpy())
        test_labels.extend(test_batch_labels.cpu().numpy())
        pred_labels.extend(test_y_.cpu().numpy())
        
test_acc = float(test_num_correct) / test_total_samples
test_auc = roc_auc_score(test_labels, test_probs)
precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
test_aupr = auc(recall, precision) 
confusion = confusion_matrix(test_labels, pred_labels)           

TP = confusion[1, 1]  # True Positives
FP = confusion[0, 1]  # False Positives
TN = confusion[0, 0]  # True Negatives
FN = confusion[1, 0]  # False Negatives

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1_score = 2 * (precision * recall) / (precision + recall)
print(confusion)

# Save prediction results
flat_list = [item for sublist in test_probs for item in sublist]
np.savetxt('result/test_data_prediction.txt', np.array(flat_list), fmt='%.2f',delimiter='\t')    

print('test_data_AUROC:', np.mean(test_auc))
print('test_data_AUPR:' , np.mean(test_aupr))

