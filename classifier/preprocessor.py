import pandas as pd
import numpy as np
import re, os
from datetime import datetime
from bounter import bounter
from konlpy.tag import Mecab

class Preprocessor():
    """
    Tagging and padding data
    """
    def __init__(self, 
                 input_path,
                 max_len,
                 min_len,
                 use_min_cnt,
                 word_min_cnt,
                 ):
        self.df = pd.read_table(input_path)
        self.max_len, self.min_len = max_len, min_len
        self.use_min_cnt = use_min_cnt
        self.word_min_cnt = word_min_cnt

        self.tagger = Mecab()

    def preprocess(self):
        """
        """
        def _tag(df):
            df = df.assign(
                processed = df['document'].apply(lambda x :''.join(re.findall('[가-힣\s0-9]',str(x)))),
            ).dropna()
            df = df.assign(
                mecab = df['processed'].apply(lambda x: ['/'.join(wp) for wp in self.tagger.pos(x)])
            )
            return df.assign(
                mecab_len = df['mecab'].apply(lambda x: len(x))
            )

        def _cut_by_len(df):
            return df.loc[lambda x:(x.mecab_len<=self.max_len) & (x.mecab_len>=self.min_len)]

        def _index_words(df):
            all_words = np.concatenate(df['mecab'].values)

            if self.use_min_cnt:
                cnt = bounter(size_mb=4096)
                cnt.update(all_words)
                words_cnt = np.array(list(cnt.iteritems()))
                words = words_cnt[:,0]
                cnts = words_cnt[:,1]
                cnts = cnts.astype(int)
                unique_words = words[np.where(cnts >= self.word_min_cnt)]
            else:
                unique_words = pd.unique(all_words)
            print('number of unique_words:{}'.format(unique_words.shape[0]))

            w2i, i2w = {}, {}
            for i, w in enumerate(unique_words):
                w2i[w] = i+1
                i2w[i+1] = w

            return unique_words, w2i, i2w

        def _drop_pad(df):
            def _drop_absent(li):
                try:
                    return [self.w2i[w] for w in li]
                except KeyError:
                    return None

            df = df.assign(
                mecab_index = df['mecab'].apply(_drop_absent)
            ).dropna()
            return df.assign(
                padded = df['mecab_index'].apply(lambda x: np.pad(x, [0,self.max_len-len(x)], 'constant'))
            )

        df = _tag(self.df)
        df = _cut_by_len(df)
        self.unique_words, self.w2i, self.i2w = _index_words(df)
        df = _drop_pad(df)
        self.data = df.loc[:,['padded', 'label']]
        return self.data

    def save(self, dir_path, tag=''):
        name = '{}_{}_{}_{}_{}'.format(self.max_len,self.min_len,self.use_min_cnt,self.word_min_cnt,tag)
        save_path = os.path.join(dir_path,name)
        if name not in os.listdir(dir_path):
            os.mkdir(save_path)
        self.data.to_csv(os.path.join(save_path, 'data.csv'), index=False)
        for obj in ['w2i', 'i2w', 'unique_words']:
            np.save(os.path.join(save_path, obj), getattr(self, obj))