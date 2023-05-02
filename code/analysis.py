from text_mining import TextMining
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

numPaperInDoc=10
start=2
stop=21
step=1

coherence_vals = [] # 一致性
perplexity_vals = [] # 困惑性

for n_topic in range(start,stop,step):
    lda_model,dictionary,corpus,texts = TextMining(data_folder='/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/data',
               result_folder=f'/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/result/result_numTopic{n_topic}_numPaperInDoc{numPaperInDoc}',
               num_topics=n_topic,
               numPaperInDoc=numPaperInDoc) # 主题挖掘
    perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus))) # 计算困惑性
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='u_mass') # 计算一致性
    coherence_vals.append(coherence_model_lda.get_coherence())

# 绘图
x = range(start, stop, step) # 横坐标
fig, ax1 = plt.subplots(figsize=(12,5))
# 一致性
c1 = 'darkturquoise'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)
# 困惑性
c2 = 'slategray'
ax2 = ax1.twinx()
ax2.plot(x, perplexity_vals, 'o-', color=c2)
ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)
# 可视化
ax1.set_xticks(x)
fig.tight_layout()
plt.savefig(f'/home/ubuntu/mwz/Spatio-temporal_data_mining_and_analysis/Text-Mining/analysis/analysis_numTopic_numPaperInDoc{numPaperInDoc}.png')