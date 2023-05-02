# date : 2023-4-25
# language : python
# version : 3.9.7
# coder : mwz
# email : mawenzhuogorgeous@gmail.com
# topic : text mining
# details : Please use text mining method or network mining method according to Article Title (optional), Author Keywords, Keywords Plus, Abstract (optional)
#           and other fields to find the main research direction in the field of data mining, and briefly describe; Which directions are promising, and why?
# relative link: https://www.codenong.com/0d03ea499b88abf56819/

from gensim import corpora,models
import pandas as pd
import os,re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim_models
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
    data_sum.sort(key=lambda x:str(x[2])) # 按发表年份排序
    return data_sum

def dataPrepoccess(data,numPaperInDoc):
    '''
    数据预处理,组成database
    params:
        data:处理数据
        numPaperInDoc:由几篇文章构成一个文档
    returns:
        database:处理后的数据
    '''
    database=[]
    stopwords=['data mining']
    keys_cleaned=[]
    for i in range(len(data)):
        # 全部小写化
        data[i][0]=data[i][0].lower()
        data[i][1]=data[i][1].lower()
        # 分割关键词
        a=data[i][0].split('; ') # 分割Author Keywords字段
        b=data[i][1].split('; ') # 分割Keywords Plus字段
        keys=a+b
        # 去除‘’
        if a==['']:
            keys.remove('')
        if b==['']:
            keys.remove('')
        # 数据清理
        for key in keys:
            if len(key)<3: # 删除长度太短的字符
                continue
            if key in stopwords: # 删除停用词
                continue
            key=re.sub(r'[^A-Za-z0-9,\s\-]+','',key) # 替换特殊字符
            keys_cleaned.append(key)
        if numPaperInDoc==0: # 同年文章构成一个文档
            if i!=len(data)-1 and data[i][2]!=data[i+1][2]: # 下个paper的发布年份和现在这个paper不一致
                database.append(keys_cleaned) # 纳入数据库
                keys_cleaned=[] 
        else:
            if (i+1)%numPaperInDoc==0: # 每numPaperInDoc篇文章构成一个document
                database.append(keys_cleaned) # 纳入数据库
                keys_cleaned=[]
    return database

def createWordCloud(lda_model,image_path):
    '''
    绘制词云
    params:
        lda_model: 训练好的lda模型
        image_path: 词云图片保存路径
    '''
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
    plt.savefig(image_path)# 保存图片

def TextMining(data_folder,result_folder,num_topics,numPaperInDoc=1):
    '''
    文本挖掘
    params:
        data_folder: 数据所在文件夹位置
        result_folder: 结果存放文件夹位置
        num_topics: 挖掘主题的数量
        numPaperInDoc: 每个文档由几篇文章构成
    returns:
        lda_model: 训练好的lda模型
        dictionary: 数据字典
        corpus: 词袋
        database: 原始文本
    '''
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    corpus_path=result_folder+f"/corpus_numPaperInDoc{numPaperInDoc}.pkl"
    dictionary_path=result_folder+f"/dictionary_numPaperInDoc{numPaperInDoc}.txt"
    lda_path=result_folder+f'/lda_numTopic{num_topics}_numPaperInDoc{numPaperInDoc}.model'
    topic_path=result_folder+f'/topic_numTopic{num_topics}_numPaperInDoc{numPaperInDoc}.txt'
    image_path=result_folder+f'/topic_numTopic{num_topics}_numPaperInDoc{numPaperInDoc}.png'
    pcoa_path=result_folder+f'/pcoa_numTopic{num_topics}_numPaperInDoc{numPaperInDoc}.html'
    if not os.path.exists(dictionary_path) or not os.path.exists(corpus_path) or not os.path.exists(lda_path): # 如果之前数据不存在
        print('之前不存在数据,需生成...')
        # 数据读取与预处理
        database=readFile(data_folder,['Author Keywords','Keywords Plus','Publication Year']) # 读关键词
        database=dataPrepoccess(database,numPaperInDoc) # 处理关键词字段，得到关键词库
        dictionary=corpora.Dictionary(database) # 生成字典
        dictionary.save_as_text(dictionary_path) # 保存字典
        # 生成词袋bag of word    
        corpus=[dictionary.doc2bow(text) for text in database]
        with open(corpus_path, 'wb') as f:
                pickle.dump(corpus, f) # 保存corpus
        # 使用 LDA 模型进行主题分析
        lda_model = models.LdaModel(
            corpus, # 词袋
            num_topics=num_topics, # 挖掘出的主题个数
            id2word=dictionary, # 字典 
            passes=10) # 训练过程中穿过语料库的次数
        lda_model.save(lda_path) # 保存模型
    else: # 之前保存过数据
        print('已有数据,直接导入...')
        # 数据导入
        with open(corpus_path, 'rb') as f: # 从文件中加载database变量
            corpus = pickle.load(f)
        dictionary = corpora.Dictionary.load_from_text(dictionary_path) # 导入字典数据
        lda_model = models.ldamodel.LdaModel.load(lda_path)# 加载lda模型

    # 保存主题挖掘结果
    with open(topic_path, 'w') as f:
        # 循环写入每个主题下的词语
        for topic in lda_model.print_topics():
            f.write(str(topic) + '\n')
    # 可视化
    createWordCloud(lda_model,image_path) # 绘制每个主题的词云
    # 分析
    vis_pcoa = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)# 主坐标PCoA分析
    pyLDAvis.save_html(vis_pcoa, pcoa_path) # 存为html文件
    return lda_model,dictionary,corpus,database


    



if __name__=='__main__':
    num_topics=10
    numPaperInDoc=10
    TextMining(data_folder='/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/data',
               result_folder=f'/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/result/result_numTopic{num_topics}_numPaperInDoc{numPaperInDoc}',
               num_topics=num_topics,
               numPaperInDoc=numPaperInDoc)
