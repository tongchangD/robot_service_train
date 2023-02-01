import os
from transformers import BertTokenizer
import torch


class PairData(torch.utils.data.Dataset):
    def __init__(self, opt):
        # bert
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(opt.pretrain_path, 'bert-base-chinese-vocab.txt'))
        # xlnet
        # self.tokenizer = XLNetTokenizer.from_pretrained(os.path.join(opt.pretrain_path, 'spiece.model'))
        # albert
        # self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(opt.pretrain_path, 'bert-base-multilingual-cased-vocab.txt'))
        self.x_list = []
        self.y_list = []
        self.segment = []
        self.posi = []

        data_path = os.path.join(opt.data_path, 'Q_Q.txt')
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')

                word_all = [101]  # [CLS]
                for data in line[:2]:
                    word_l = self.tokenizer.encode(data, add_special_tokens=False)
                    if len(word_l) < opt.max_len:
                        word_l = word_l + [0]*(opt.max_len - len(word_l)) # 不足补[PAD] = 0
                    else:
                        word_l = word_l[:opt.max_len]
                    word_all = word_all + word_l + [102] # 添加[SEP]

                self.x_list.append(torch.tensor(word_all))
                self.y_list.append(torch.tensor(int(line[2])))

                self.segment.append(torch.tensor([0]*(opt.max_len+2) + [1]*(opt.max_len+1)))
                self.posi.append(torch.tensor([i for i in range(len(word_all))]))


        
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index], self.segment[index], self.posi[index]
             
    def __len__(self):
        return len(self.x_list)
