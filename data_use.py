import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



def data_read(f):
    '''直接读取原始数据'''
    train_list = []
    for line in f.readlines():
        print(line)
        if line == '':
            continue
        train_list.append(line)

    train_list = np.array(train_list)
    f.close()
    return(train_list)

def get_examples(f):
    """处理数据格式"""
    dev_data = []
    index = 0
    for line in f.readlines():
        guid = index
        line = line.replace('\n', '').split('_!_')
        type_news = str(line[2])
        place = str(line[4])
        text_a = str(line[3])
        # print('label: ', line[1])
        label = str(line[1])
        dev_data.append([guid,text_a,label, type_news, place])
        index += 1
    dev_data = pd.DataFrame(dev_data, columns=['id', 'text', 'index', 'type_news', 'label'])
    return dev_data

def label_change(label):
    replace_dict = {'100': '0', '101': '1', '102': '2', '103': '3', '104': '4', '106': '5', '107': '6', '108': '7',
                    '109': '8', '110': '9', '112': '10', '113': '11', '114': '12', '115': '13', '116': '14'}
    new_lists = [replace_dict[i] if i in replace_dict else i for i in label]
    return new_lists


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels


    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


