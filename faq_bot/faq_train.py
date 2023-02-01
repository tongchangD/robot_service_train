from transformers import BertForSequenceClassification
import torch
import time
import argparse
from faq_bot.PairData import PairData
from faq_bot.saver import loadConversations

import lucene, os, sys, time, threading
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)
class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir, analyzer):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 100)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer)
        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    def indexDocs(self, root, writer):
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(False)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        for root, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith('.txt'):
                    continue
                # print ("adding", filename)
                try:
                    path = os.path.join(root, filename)
                    file = open(path)
                    # contents = unicode(file.read(), 'iso-8859-1')
                    contents = file.read()
                    file.close()
                    doc = Document()
                    doc.add(Field("name", filename, t1))
                    doc.add(Field("path", root, t1))
                    if len(contents) > 0:
                        doc.add(Field("contents", contents, t2))
                    else:
                        print ("warning: no content in %s" % filename)
                    writer.addDocument(doc)
                except Exception as e:
                    print ("Failed in indexDocs:", e)

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help = '每批数据的数量') # ss32
    parser.add_argument('--nepoch', type=int, default=6, help = '训练的轮次')
    parser.add_argument('--lr', type=float, default=1e-5, help = '学习率')
    parser.add_argument('--gpu', type=bool, default=True, help = '是否使用gpu')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader使用的线程数量')
    parser.add_argument('--num_labels', type=int, default=2, help='分类类数')
    parser.add_argument('--data_path', type=str, default='./faq_bot/data', help='数据路径')
    parser.add_argument('--pretrain_path', type=str, default='./faq_bot/pre_model', help='预训练模型路径')
    parser.add_argument('--max_len', type=str, default=64, help='每个str的最大长度')
    parser.add_argument('--version', type=float, default=2.2, help='版本2.1表示只有一个显示答案')
    parser.add_argument('--save_dir', type=str, default='./faq_bot/outs', help='保存地址')
    opt=parser.parse_args()
    print(opt)
    return opt

def get_model(opt):
    # bert
    model = BertForSequenceClassification.from_pretrained(opt.pretrain_path, num_labels=opt.num_labels)
    return model

def get_data(opt):
    trainset = PairData(opt)
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    return trainloader



    
def train(epoch, model, trainloader, optimizer, opt):
    print('\ntrain-Epoch: %d' % (epoch+1))
    model.train()
    start_time = time.time()
    print_step = 1
    for batch_idx,(data, label, segment, posi) in enumerate(trainloader):
        if opt.gpu:
            data = data.cuda()
            segment = segment.cuda()
            posi = posi.cuda()
            label = label.unsqueeze(1).cuda()
        
        
        optimizer.zero_grad()
        # bert
        outputs = model(data, token_type_ids=segment, position_ids=posi,labels = label)

        loss, logits = outputs[0],outputs[1]
        loss.backward()
        optimizer.step()
        
        if batch_idx % print_step == 0:
            print("Epoch:%d [%d|%d] loss:%f" %(epoch+1,batch_idx,len(trainloader),loss.mean()))
    print("time:%.3f" % (time.time() - start_time))


def test(epoch,model,trainloader,testloader,opt):
    print('\ntest-Epoch: %d' % (epoch+1))
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for batch_idx,(data, label, segment, posi) in enumerate(testloader):
            if opt.gpu:
                data = data.cuda()
                segment = segment.cuda()
                posi = posi.cuda()
                labels = label.unsqueeze(1).cuda()
            else:
                labels = label.unsqueeze(1)

            # bert
            outputs = model(data, token_type_ids=segment, position_ids=posi, labels=labels)
            # xlnet
            # outputs = model(data, labels=labels)
            loss, logits = outputs[:2]
            _,predicted=torch.max(logits.data,1)


            total+=data.size(0)
            correct+=predicted.data.eq(labels.squeeze().data).cpu().sum()
    
    s = ("Acc:%.3f" %((1.0*correct.numpy())/total))
    print(s)



def train_faq():
    opt = get_train_args()
    model = get_model(opt)
    trainloader = get_data(opt)

    if opt.gpu:
        model.cuda()
    # optimizer = torch.optim.AdamW(model.parameters(), )
    # optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr, momentum=0.9)
    optimizer=torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer=torch.optim.AdamW(model.parameters(), lr=opt.lr) # torch 1.3 以上
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.exists(os.path.join(opt.save_dir, 'model.pth')):
        for epoch in range(opt.nepoch):
            train(epoch,model,trainloader,optimizer,opt)
        torch.save(model.state_dict(),os.path.join(opt.save_dir, 'model.pth'))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.save_dir, 'model.pth')))
        print('模型存在,直接test')
        # test(0,model,trainloader,trainloader,opt)

    ######保存训练和测试

    save_txt = loadConversations(opt, 'ourdata', 'ID+答案')
    # 训练pylucene

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    try:
        # IndexFiles(save_txt, INDEX_DIR, WhitespaceAnalyzer())
        IndexFiles(save_txt, os.path.join(opt.save_dir, 'IndexFiles.index'), StandardAnalyzer())
    except Exception as e:
        print("Failed: ", e)
        raise e


       
