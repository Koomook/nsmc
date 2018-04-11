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
                 input_path=None,
                 max_len=None,
                 min_len=None,
                 use_min_cnt=None,
                 word_min_cnt=None,
                 load_preprocessed=False,
                 dir_path=None
                 ):
        if not load_preprocessed:
            """
            This must be changed
            """
            self.df = pd.read_table(input_path)
            self.max_len, self.min_len = max_len, min_len
            self.use_min_cnt = use_min_cnt
            self.word_min_cnt = word_min_cnt

            self.tagger = Mecab()
        else:
            self.load(dir_path)

    def _tag(self, df):
        df = df.assign(
            processed = df['document'].apply(lambda x :''.join(re.findall('[가-힣\s0-9]',str(x)))),
        ).dropna()
        df = df.assign(
            mecab = df['processed'].apply(lambda x: ['/'.join(wp) for wp in self.tagger.pos(x)])
        )
        return df.assign(
            mecab_len = df['mecab'].apply(lambda x: len(x))
        )

    def _cut_by_len(self, df):
        return df.loc[lambda x:(x.mecab_len<=self.max_len) & (x.mecab_len>=self.min_len)]

    def _drop_pad(self, df):
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

    def preprocess_train(self):
        """
        """
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

        df = self._tag(self.df)
        df = self._cut_by_len(df)
        self.unique_words, self.w2i, self.i2w = _index_words(df)
        df = self._drop_pad(df)
        self.data = df.loc[:,['padded', 'label']]
        return self.data

    def preprocess_test(self, train_path):
        self.load(train_path)
        df = self._tag(self.df)
        df = self._cut_by_len(df)
        df = self._drop_pad(df)
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

    def load(self, dir_path):
        for f in os.listdir(dir_path):
            if re.search('npy', f):
                setattr(self, f[:-4], np.load(os.path.join(dir_path, f)))
        self.data = pd.read_csv(os.path.join(dir_path, 'data.csv'))

    def generate_batch(self, batch_size):
        while True:
            batch_idx = np.random.randint(0, self.data.shape[0], size=batch_size)
            batch_data = self.data.values[batch_idx]
            batch_inputs =  np.concatenate(batch_data[:,0]).reshape([batch_size,-1])
            batch_targets = np.expand_dims(batch_data[:,1], axis=1)
            yield batch_inputs, batch_targets