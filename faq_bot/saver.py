import pickle
import os

def loadConversations(config, conv_file, id_file):
    # 保存答案
    # f_response = open(self.dir + '/responses', 'w')
    # f_que = open(self.dir + '/questions', 'w')
    save_txt = './faq_bot/txt'
    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
    f_response = []
    f_que = []
    f_con = open(os.path.join(config.data_path, conv_file))
    lines = f_con.readlines()
    # queries = []
    index = 0
    i = 0
    for content in lines:
        # content = content.lower()
        # 问题 经过切词，保存为list
        if index % 3 != 0:
            content = content[2:]
        if index % 3 == 1:
            content = content.lower()
            if '\ufeff' in content:
                content = content[1:]
            # f_que.write(content)
            content = content[:-1]
            f_que.append(content)

            f_txt = open(save_txt + '/' + str(i) + '.txt', 'w')
            f_txt.write(content)
            f_txt.close()
            i += 1
        # 答案保存到txtz中
        elif index % 3 == 2:
            if '\ufeff' in content:
                content = content[1:]
            # f_response.write(content)
            f_response.append(content[:-1])
        index += 1
    # f_response.close()
    # f_que.close()
    # 答案 和 id
    file1 = open(os.path.join(config.data_path, id_file), 'r')
    lines = file1.readlines()
    dict = {}
    for line in lines:
        # print(line)
        line = line[:-1]
        linelist = line.split('\t')
        if config.version == 2.1:
            dict[linelist[1]] = linelist[0] # v2.1
        else:
            dict[linelist[1]] = [linelist[0], linelist[2]] #v2.2

    id_key = {}
    for i in f_response:
        if i in dict.keys() and i not in id_key.keys():
            id_key[i] = dict[i]
        if i not in dict.keys():
            print(i)

    filename = config.save_dir + '/conversion.pkl'
    with open(filename, 'wb') as handle:
        data = {
            'question': f_que,
            'response': f_response,
            'id': id_key,
            # 'oriTrainingSamples':self.oriTrainingSamples
        }

        pickle.dump(data, handle, -1)  # Using the highest protocol available

    return save_txt
