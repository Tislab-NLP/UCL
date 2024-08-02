import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
import sklearn
import torch.nn.functional as F
import torch
import torch.nn as nn
import transformers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer,BertConfig,AdamW,AlbertTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score
import matplotlib.pyplot as plt
from SCL import *
from sklearn.metrics import confusion_matrix

from load import *
from metrics import *
from metrics1 import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#设定参数
batch_size = 64
tem = 0.1
tem1 = 0.07
lam = 0.6
fs = 0.7
total_epochs = 3
lr = 1e-5
save_model_path = '/root/autodl-tmp/FSCL/save/en/c1bert1.pkl'

ytrue = pd.read_csv('/root/autodl-tmp/FSCL/data/en/test.csv',usecols=['label'])
strue = pd.read_csv('/root/autodl-tmp/FSCL/data/en/test.csv',usecols=['race'])
strue = strue.values

train_data_path="/root/autodl-tmp/FSCL/data/en/train.csv"
dev_data_path="/root/autodl-tmp/FSCL/data/en/dev.csv"
test_data_path="/root/autodl-tmp/FSCL/data/en/test.csv"#调用load_data函数，将数据加载为Tensor形式
train_data = load_traindata(train_data_path)
#train_data = F_load_traindata(train_data_path)


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        # 加载预训练模型
        pretrained_weights = "/root/autodl-tmp/CSCL4FTC/save_uncased/one"
        self.bert = transformers.BertModel.from_pretrained(pretrained_weights,output_attentions=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义线性函数
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 得到bert_output
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 获得预训练模型的输出
        bert_cls_hidden_state = bert_output[1]
        # 将768维的向量输入到线性层映射为二维向量
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output,bert_cls_hidden_state



#引入数据路径


dev_data = load_devdata(dev_data_path)
test_data = load_testdata(test_data_path)
#将训练数据和测试数据进行DataLoader实例化
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def dev(model,dev_loader):
    #将模型放到服务器上
    model.to(device)
#设定模式为验证模式
    model.eval()
#设定不会有梯度的改变仅作验证
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids,token_type_ids,attention_mask,labels) in tqdm(enumerate(dev_loader),desc='Dev Itreation:'):

            input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)

            out_put,hiden_state = model(input_ids,token_type_ids,attention_mask)

            _, predict = torch.max(out_put.data, 1)

            correct += (predict==labels).sum().item()

            total += labels.size(0)

        res = correct / total
        return res

def train(model,train_loader,dev_loader,tem,lam,tem1) :
    #将model放到服务器上
    model.to(device)
    #设定模型的模式为训练模式
    model.train()
    #定义模型的损失函数
    criterion = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #学习率的设置
    optimizer_params = {'lr': lr, 'eps': 1e-6, 'correct_bias': False}
    #使用AdamW 主流优化器
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    #学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)
    t_total = len(train_loader)
    #设定训练轮次

    bestAcc = 0
    correct = 0
    total = 0
    print('Training and verification begin!')
    for epoch in range(total_epochs):
        #for step, (input_ids, token_type_ids, attention_mask, labels,s_labels) in enumerate(train_loader):
        for step, (input_ids,token_type_ids,attention_mask,labels) in enumerate(train_loader):
#从实例化的DataLoader中取出数据，并通过 .to(device)将数据部署到服务器上
            #input_ids,token_type_ids,attention_mask,labels,s_labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device),s_labels.to(device)
            input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)
            #梯度清零
            optimizer.zero_grad()
            #将数据输入到模型中获得输出
            out_put, hiden_state =  model(input_ids,token_type_ids,attention_mask)
            #计算损失
            cross_loss = criterion(out_put, labels)
            #contrastive_lf = F_contrastive_loss(tem,hiden_state.cpu().detach().numpy(), labels,s_labels)
            #contrastive_lc = contrastive_loss(tem,hiden_state.cpu().detach().numpy(), labels)
            #contrastive_ls = Fs_contrastive_loss(tem1,hiden_state.cpu().detach().numpy(), labels,s_labels)
            #SCLoss = (lam * contrastive_lc) + (1 - lam) * cross_loss + fs * contrastive_ls
            _, predict = torch.max(out_put.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            cross_loss.backward()
            #SCLoss.backward()
            optimizer.step()
             #每两步进行一次打印
            if (step + 1) % 10 == 0:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100))
            #每五十次进行一次验证
            if (step + 1) %  400== 0:
                train_acc = correct / total
                #调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc = dev(model, dev_loader)
                if bestAcc < acc:
                    bestAcc = acc
                    #模型保存路径

                    torch.save(model, save_model_path)
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100,bestAcc*100,acc*100))
        scheduler.step(bestAcc)

#实例化模型
model = BertClassificationModel()
#调用训练函数进行训练与验证
train(model,train_loader,dev_loader,tem,lam,tem1)

def predict(model, test_loader):
    model.to(device)
    model.eval()
    predicts = []
    predict_probs = []
    with torch.no_grad():
       # print('zhunbei')
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels,s_labels) in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, labels,s_labels = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), labels.to(device),s_labels.to(device)
            #print('diyi')
            out_put,hiden_state = model(input_ids, token_type_ids, attention_mask)
            #print('zhihou')

            _, predict = torch.max(out_put.data, 1)

            pre_numpy = predict.cpu().numpy().tolist()
            predicts.extend(pre_numpy)
            probs = F.softmax(out_put).detach().cpu().numpy().tolist()
            predict_probs.extend(probs)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        test_accuracy = np.mean(res)
       #return test_accuracy,predict_probs
        print('预测准确度：',test_accuracy)
        # 返回预测结果和预测的概率
        return predicts, predict_probs



# 引进训练好的模型进行测试

Trained_model = torch.load(save_model_path)

# predicts是预测的（0   1），predict_probs是概率值
predicts,predict_probs = predict(Trained_model, test_loader)
#print(predict_probs)
#print(predicts)
print('---------------------------')



acc = accuracy_score(ytrue, predicts)
R = recall_score(ytrue, predicts)
F1 = f1_score(ytrue, predicts)


AUC = roc_auc_score(ytrue,predicts)
y_type, y_true, y_pred = sklearn.metrics._classification._check_targets(ytrue, predicts)
FPR1 = FPR(y_true,y_pred)
FNR1 = FNR(y_true,y_pred)

print('p:', acc, 'r:', R, 'F1:', F1,'AUC:',AUC,'FPR:',FPR1,'FNR:',FNR1)
print('---------------------------')
# #只有男女
# FPR_m = FPR_m(y_true,y_pred,strue)
# FNR_m = FNR_m(y_true,y_pred,strue)
# FPR_f = FPR_f(y_true,y_pred,strue)
# FNR_f = FNR_f(y_true,y_pred,strue)
# # print('FPR_m:',FPR_m,'FNR_m:',FNR_m)
# # print('FPR_f:',FPR_f,'FNR_f:',FNR_f)
# # print('---------------------------')
# FPED_B = FPED_B(FPR_m,FPR_f,FPR1)
# FNED_B = FNED_B(FNR_m,FNR_f,FNR1)
# FPED_P = FPED_P(FPR_m,FPR_f)
# FNED_P = FNED_P(FNR_m,FNR_f)
# b = FPED_B+FNED_B

# print('FPED_B:',FPED_B,'FNED_B:',FNED_B,'sum:',b)
# print('FPED_P:',FPED_P,'FNED_P:',FNED_P)
# print('---------------------------')




#四个子组都有 黑人、白人、亚裔、西班牙裔
FPR_m = FPR_m(y_true,y_pred,strue)
FNR_m = FNR_m(y_true,y_pred,strue)
FPR_f = FPR_f(y_true,y_pred,strue)
FNR_f = FNR_f(y_true,y_pred,strue)
FPR_mf = FPR_mf(y_true,y_pred,strue)
FNR_mf = FNR_mf(y_true,y_pred,strue)
FPR_3 = FPR_3(y_true,y_pred,strue)
FNR_3 = FNR_3(y_true,y_pred,strue)
# print('FPR_mf:',FPR_mf,'FNR_mf:',FNR_mf)
# print('FPR_3:',FPR_3,'FNR_3:',FNR_3)
# print('---------------------------')


FPED_B1 = FPED_B1(FPR_m,FPR_f,FPR_mf,FPR_3,FPR1)
FNED_B1 = FNED_B1(FNR_m,FNR_f,FNR_mf,FNR_3,FNR1)
FPED_P1 = FPED_P1(FPR_m,FPR_f,FPR_mf,FPR_3)
FNED_P1 = FNED_P1(FNR_m,FNR_f,FNR_mf,FNR_3)
b = FPED_B1+FNED_B1
p=FPED_P1+FNED_P1
print('FPED_B1:',FPED_B1,'FNED_B1:',FNED_B1,'sum:',b)
print('FPED_P1:',FPED_P1,'FNED_P1:',FNED_P1,'sum:',p)





