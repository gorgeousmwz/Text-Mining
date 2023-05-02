# date : 2023-4-25
# language : python
# version : 3.9.7
# coder : mwz
# email : mawenzhuogorgeous@gmail.com
# topic : text mining
# details : Please use text mining method or network mining method according to Article Title (optional), Author Keywords, Keywords Plus, Abstract (optional)
#           and other fields to find the main research direction in the field of data mining, and briefly describe; Which directions are promising, and why?
# relative link: https://www.codenong.com/0d03ea499b88abf56819/

import nltk
from nltk.corpus import stopwords
from gensim import corpora,models
import pandas as pd
import os,re
import matplotlib.pyplot as plt
import prettytable as pt
from wordcloud import WordCloud
import pyLDAvis
import math
import pickle


def readFile(path,constraint=None,no_values=''):
    '''
    读取数据文件,返回list
    params:
        path: 数据文件夹路径
        constraint: 读取指定列的列名称列表
        no_values: 空缺值处理
    returns:
        data_sum:数据列表
    '''
    files=os.listdir(path)
    data_sum=[]
    for filename in files:
        file=os.path.join(path,filename)
        data=pd.read_excel(io=file,usecols=constraint,engine='xlrd') # 读取文件
        data.fillna(no_values,inplace=True) # 填写空缺值
        data=data.values.tolist() # 转为list
        data_sum+=data
    return data_sum#[:100]

def dataPrepoccess(data,object):
    '''
    数据预处理,组成database
    params:
        data:处理数据
        object:处理对象('keyword' or 'title+abstract')
    returns:
        database:处理后的数据
    '''
    database=[]
    if object=='keyword': # 对关键词的处理
        for i in range(len(data)):
            a=data[i][1].split('; ') # 分割Author Keywords字段
            b=data[i][2].split('; ') # 分割Keywords Plus字段
            for j in range(len(a)): # 全部转小写
                a[j]=a[j].lower()
            for j in range(len(b)):
                b[j]=b[j].lower()
            keys=a+b
            # 去除‘’
            if a==['']:
                keys.remove('')
            if b==['']:
                keys.remove('')
            data[i][0]=re.sub(r'[^\w\s]', '', data[i][2]) # 去除符号
            data[i][3]=re.sub(r'[^\w\s]', '', data[i][3]) # 去除符号
            data[i][0]=data[i][0].split(' ') # 分割单词
            data[i][3]=data[i][3].split(' ') # 分割单词
            words=data[i][0]+data[i][3] # 合并title和abstract字段单词
            words=dataClean(words) # 数据清洗
            # 删除title和abstract分割出的单词中在关键词中出现过的词
            for key in keys:
                key=key.split(' ')
                for i in key:
                    if i in words:
                        words.remove(i)
            database.append(keys+words) # 纳入数据库
    elif object=='title+abstract': # 对文章标题和摘要的处理
        for item in data:
            for i in range(len(item)):
                item[i]=re.sub(r'[^\w\s]', '', item[i]) # 去除符号
                item[i]=item[i].split(' ') # 分割单词
                item[i]=dataClean(item[i]) # 数据清洗
            database.append(item[0]+item[1]) # 纳入数据库
    return database

def dataClean(data):
    '''
    数据清洗:清除停用词、短词
    '''
    stop_words=stopwords.words('english') # 获取nltk停用词库
    word_index=0
    while word_index<len(data):
        data[word_index]=data[word_index].lower() # 全部取小写
        if len(data[word_index]) <3: # 删除短词
            data.pop(word_index)
            continue
        if data[word_index] in stop_words: # 删除停用词
            data.pop(word_index)
            continue
        word_index+=1
    return data

def createWordCloud(lda_model):
    fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda_model.num_topics/2), figsize=(16,20))
    axs = axs.flatten()

    def color_func(word, font_size, position, orientation, random_state, font_path):
        return 'darkturquoise'

    for i, t in enumerate(range(lda_model.num_topics)):

        x = dict(lda_model.show_topic(t, 30))
        im = WordCloud(
            background_color='black',
            color_func=color_func,
            max_words=4000,
            width=300, height=300,
            random_state=0
        ).generate_from_frequencies(x)
        axs[i].imshow(im.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
        axs[i].axis('off')
        axs[i].set_title('Topic '+str(t))

    # vis
    plt.tight_layout()
    #plt.show()

    # save as png
    plt.savefig('/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/result/result+_numTopic10_numPaperInDoc1/wordcloud+.png')


def TextMining(data_folder,num_topics):
    '''
    文本挖掘
    '''
    nltk.data.path.insert(0,'/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/nltk_data') # 指定nltk库数据位置
    database_path="/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/result/result+_numTopic10_numPaperInDoc1/database+.pkl"
    dictionary_path="/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/result/result+_numTopic10_numPaperInDoc1/dictionary+.txt"
    if not os.path.exists(dictionary_path) or not os.path.exists(database_path): # 如果之前数据不存在
        print('之前不存在数据,需生成...')
        # 数据读取与预处理
        database=readFile(data_folder,['Article Title','Author Keywords','Keywords Plus','Abstract']) # 读关键词
        database=dataPrepoccess(database,'keyword') # 处理关键词字段，得到关键词库
        # database=readFile(data_folder,['Article Title','Abstract']) # 读题目
        # database=dataPrepoccess(database,'title+abstract') # 处理题目字段，得到题目库
        dictionary=corpora.Dictionary(database) # 生成字典
        # 数据存储
        with open(database_path, 'wb') as f:
            pickle.dump(database, f) # 将变量database存储到文件中
        dictionary.save_as_text(dictionary_path) # 保存字典
    else: # 之前保存过数据
        print('已有数据,直接导入...')
        # 数据导入
        with open(database_path, 'rb') as f: # 从文件中加载database变量
            database = pickle.load(f)
        dictionary = corpora.Dictionary.load_from_text(dictionary_path) # 导入字典数据

    # 生成词袋bag of word    
    corpus=[dictionary.doc2bow(text) for text in database]
    # 使用 LDA 模型进行主题分析
    lda_model = models.LdaModel(
        corpus, # 词袋
        num_topics=num_topics, # 挖掘出的主题个数
        id2word=dictionary, # 词带 
        passes=10) # 训练过程中穿过语料库的次数

    # 打印每个主题下的前几个词语
    for topic in lda_model.print_topics():
        print(topic)

    createWordCloud(lda_model)


    



if __name__=='__main__':
    TextMining('/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/data',10)
