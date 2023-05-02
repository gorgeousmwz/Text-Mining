topic : Text Mining & Topic analysis
coder : Wenzhuo Ma
email : mawenzhuogorgeous@gmail.com
date : 2023-04-30
language : python
version : 3.7.16
third-party library : gensim、nltk、wordcloud、pyLDAvis、pandas、os、re、pickle
details : Please use text mining method or network mining method according to Article Title (optional), Author Keywords, Keywords Plus, Abstract (optional)
          and other fields to find the main research direction in the field of data mining, and briefly describe; Which directions are promising, and why?


file structure:
    analysis:分析结果
        analysis_numTopic_numPaperInDoc{0|1|10}.png: 不同numPaperInDoc的主题数量分析图(一致性和困惑性)
    code:项目源码
        text_mining.py: 项目主要代码，文本挖掘
        text_mining_try.py: 添加了Paper Title和Abstract字段的文本挖掘
        install_nltk.py: 安装nltk代码
        analysis.py: 模型分析代码(分析不同numTopic)
    result:挖掘结果
        result_numTopic{2-20}_numPaperInDoc{0|1|10}: 对应numTopic和numPaperInDoc的挖掘结果文件夹
            corpus_numTopic_numPaperInDoc.pkl：词袋变量
            dictionary_numTopic_numPaperInDoc.txt：数据字典文本
            lda_numTopic_numPaperInDoc.model(.npy,.id2word,.state)：LDA模型
            pcoa_numTopic_numPaperInDoc.html：主题PCoA分析结果网页
            topic_numTopic_numPaperInDoc.png：主题词云图
            topic_numTopic_numPaperInDoc.txt：主题输出文本
    步骤流程图.jpg
    分析报告.pdf
    readme.txt