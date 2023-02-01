# main.py
from transformers import BertTokenizer, XLNetTokenizer
from transformers import BertForSequenceClassification, XLNetForSequenceClassification

import torch
import time
import argparse
from PairData import PairData
import os


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10, help='每批数据的数量')
    parser.add_argument('--nepoch', type=int, default=3, help='训练的轮次')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用gpu')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader使用的线程数量')
    parser.add_argument('--num_labels', type=int, default=2, help='分类类数')
    parser.add_argument('--data_path', type=str, default='./data/Q_Q.txt', help='数据路径')
    parser.add_argument('--pretrain_path', type=str, default='./pre_model', help='预训练模型路径')
    parser.add_argument('--max_len', type=str, default=64, help='每个str的最大长度')
    opt = parser.parse_args()
    print(opt)
    return opt


def get_model(opt):
    # bert
    model = BertForSequenceClassification.from_pretrained(opt.pretrain_path, num_labels=opt.num_labels)
    # model = XLNetForSequenceClassification.from_pretrained(opt.pretrain_path, num_labels=opt.num_labels)
    return model


def get_data(opt):
    trainset = PairData(opt)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers)
    # testset = NewsData(opt.data_path,is_train = 0)
    # testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    return trainloader


def train(epoch, model, trainloader, optimizer, opt):
    print('\ntrain-Epoch: %d' % (epoch + 1))
    model.train()
    start_time = time.time()
    print_step = int(len(trainloader) / 10)
    for batch_idx, (data, label, segment, posi) in enumerate(trainloader):
        if opt.gpu:
            data = data.cuda()
            segment = segment.cuda()
            posi = posi.cuda()
            label = label.unsqueeze(1).cuda()

        optimizer.zero_grad()
        outputs = model(data, segment, position_ids=posi, labels=label)

        loss, logits = outputs[0], outputs[1]
        loss.backward()
        optimizer.step()

        if batch_idx % print_step == 0:
            print("Epoch:%d [%d|%d] loss:%f" % (epoch + 1, batch_idx, len(trainloader), loss.mean()))
    print("time:%.3f" % (time.time() - start_time))


def test(epoch, model, trainloader, testloader, opt):
    print('\ntest-Epoch: %d' % (epoch + 1))
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (sue, label, posi) in enumerate(testloader):
            if opt.gpu:
                sue = sue.cuda()
                posi = posi.cuda()
                labels = label.unsqueeze(1).cuda()
                label = label.cuda()
            else:
                labels = label.unsqueeze(1)

            outputs = model(sue, labels=labels)
            loss, logits = outputs[:2]
            _, predicted = torch.max(logits.data, 1)

            total += sue.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()

    s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
    print(s)


if __name__ == '__main__':
    opt = get_train_args()
    model = get_model(opt)

    if opt.gpu:
        model.cuda()

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print('模型存在,直接test')
    str1 = '早上吃什么'
    str2 = '晚上吃啥'
    segment = []
    data_l = [101]
    # Bert
    tokenizer = BertTokenizer.from_pretrained(os.path.join(opt.pretrain_path, 'bert-base-chinese-vocab.txt'))
    data_l_1 = tokenizer.encode(str1) + [102]
    segment += [0]*(len(data_l_1)+1)
    data_l += data_l_1
    #
    # data_l_2 = tokenizer.encode(str2) + [102]
    # segment += [1] * (len(data_l_2))
    # data_l += data_l_2
    #
    # pos_i = [i for i in range(len(data_l))]
    #
    # if opt.gpu:
    #     data_l = torch.tensor(data_l).cuda().unsqueeze(0)
    #     segment = torch.tensor(segment).cuda().unsqueeze(0)
    #     pos_i = torch.tensor(pos_i).cuda().unsqueeze(0)
    #     model.cuda()

    # XLNet
    tokenizer = XLNetTokenizer.from_pretrained(os.path.join(opt.pretrain_path, 'spiece.model'))
    data_l_1 = tokenizer.encode(str1, add_special_tokens=False)
    if len(data_l_1) < opt.max_len:
        data_l_1 += [0]*(opt.max_len - len(data_l_1)) + [102]
    segment += [0] * (opt.max_len + 2)
    data_l_2 = tokenizer.encode(str2)
    if len(data_l_2) < opt.max_len:
        data_l_2 += [0]*(opt.max_len - len(data_l_2)) + [102]
    segment += [1]*(opt.max_len + 1)
    data_l += data_l_1 + data_l_2

    if opt.gpu:
        data_l = torch.tensor(data_l).cuda().unsqueeze(0)
        segment = torch.tensor(segment).cuda().unsqueeze(0)
        model.cuda()



    # outputs = model(data_l, token_type_ids=segment, position_ids=pos_i)
    outputs = model(data_l, token_type_ids=segment)
    preds = torch.max(torch.softmax(outputs[0], dim=1), 1)
    print(preds)



