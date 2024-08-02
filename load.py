import csv
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer,AlbertTokenizerFast



def encoder(max_len, vocab_path, text_list):
    # 将text_list embedding成bert模型可用的输入形式
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


def load_traindata(path):
    csvFileObj = open(path, encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    s_labels = []
    for row in readerObj:
        # 跳过表头
        if readerObj.line_num == 1:
            continue
        # label在什么位置就改成对应的index
        label = int(row[4])
        text = row[5]
        #s_label = int(row[3])
        text_list.append(text)
        labels.append(label)
        #s_labels.append(s_label)
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=150,
                                                        vocab_path="D:\pre-trained model/bert-base-chinese/vocab.txt",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    #s_labels = torch.tensor(s_labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels )#, s_labels)
    return data



def F_load_traindata(path):
    csvFileObj = open(path, encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    s_labels = []
    for row in readerObj:
        # 跳过表头
        if readerObj.line_num == 1:
            continue
        # label在什么位置就改成对应的index
        label = int(row[3])
        text = row[4]
        s_label = int(row[2])
        text_list.append(text)
        labels.append(label)
        s_labels.append(s_label)
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=150,
                                                        vocab_path="D:\pre-trained model/chinese-roberta-wwm-ext/vocab.txt",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    s_labels = torch.tensor(s_labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels , s_labels)
    return data


def load_testdata(path):
    csvFileObj = open(path, encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    s_labels = []
    for row in readerObj:
        # 跳过表头
        if readerObj.line_num == 1:
            continue
        # label在什么位置就改成对应的index
        label = int(row[3])
        text = row[5]
        s_label = int(row[2])
        text_list.append(text)
        labels.append(label)
        s_labels.append(s_label)
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=150,
                                                        vocab_path="D:\pre-trained model/bert-base-chinese/vocab.txt",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    s_labels = torch.tensor(s_labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels,s_labels)
    return data


def load_devdata(path):
    csvFileObj = open(path, encoding='utf-8')
    readerObj = csv.reader(csvFileObj)
    text_list = []
    labels = []
    s_labels = []
    for row in readerObj:
        # 跳过表头
        if readerObj.line_num == 1:
            continue
        # label在什么位置就改成对应的index
        label = int(row[4])
        text = row[5]
        text_list.append(text)
        labels.append(label)
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=150,
                                                        vocab_path="D:\pre-trained model/bert-base-chinese/vocab.txt",
                                                        text_list=text_list)
    labels = torch.tensor(labels)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data