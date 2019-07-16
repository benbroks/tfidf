import math
import json
from textblob import TextBlob as tb
import re
import os
import nltk

from nltk.corpus import stopwords

class TFIDF():
    def __init__(self,load_fp=None):
        if load_fp is None:
            self.tot_freq_dict = {}
            self.doc_freq_dict = {}
            self.corpus_length = 0
            self.stop_words = set(stopwords.words('english'))
        else:
            self.load(load_fp)
            self.stop_words = set(stopwords.words('english'))

    def preprocess(self,str_):
        str_ = str_.strip()
        str_ = str_.lower()
        str_ = re.sub('&',' and ',str_)
        str_ = re.sub(r'[^\w\d\s]+', ' ', str_)
        return str_

    def batch_train_w_lists(self,str_list,id_list,verbose=False,percent_cut=0.001,clean=False,include_stop=True):
        if len(str_list) != len(id_list):
            raise Exception("Batch Training Error: Mismatched list sizes.")

        # For Verbose
        num_docs = len(str_list)

        for index, item in enumerate(str_list):
            mini_dict = {}
            if clean:
                item = self.preprocess(item)
            blob = tb(item)
            doc_length = len(blob.words)

            if include_stop:
                to_iterate = set(blob.words)
            else:
                to_iterate = set(blob.words) - self.stop_words

            for word in to_iterate:
                mini_dict[word] = blob.words.count(word) / doc_length
                if word not in self.doc_freq_dict:
                    self.doc_freq_dict[word] = 1
                else:
                    self.doc_freq_dict[word] += 1
            self.tot_freq_dict[id_list[index]] = mini_dict

            if verbose and ((index/num_docs) >= percent_cut):
                print(index/num_docs*100, "percent complete.")
                percent_cut += 0.001
        
        self.corpus_length += len(str_list)
        # for normalization
        self.max_dict = {}
        for doc_id in self.tot_freq_dict:
            max = 0
            for word in self.tot_freq_dict[doc_id]:
                spec_tfidf = self.tfidf(doc_id,word)
                if spec_tfidf > max:
                    max = spec_tfidf
            self.max_dict[doc_id] = max

    def batch_train_w_dict(self,str_dict,verbose=False,percent_cut=0.001,clean=False,include_stop=True):
        # For Verbose
        num_docs = len(str_dict)

        for index, item in enumerate(str_dict):
            mini_dict = {}
            if clean:
                str_dict[item] = self.preprocess(str_dict[item])
            blob = tb(str_dict[item])
            doc_length = len(blob.words)

            if include_stop:
                to_iterate = set(blob.words)
            else:
                to_iterate = set(blob.words) - self.stop_words

            for word in to_iterate:
                mini_dict[word] = blob.words.count(word) / doc_length
                if word not in self.doc_freq_dict:
                    self.doc_freq_dict[word] = 1
                else:
                    self.doc_freq_dict[word] += 1
            self.tot_freq_dict[item] = mini_dict

            if verbose and ((index/num_docs) >= percent_cut):
                print(index/num_docs*100, "percent complete.")
                percent_cut += 0.001
        
        self.corpus_length += len(str_dict)
        # for normalization
        self.max_dict = {}
        self.absolute_max = 0
        for doc_id in self.tot_freq_dict:
            max = 0
            for word in self.tot_freq_dict[doc_id]:
                spec_tfidf = self.tfidf(doc_id,word)
                if spec_tfidf > max:
                    max = spec_tfidf
                    if spec_tfidf > self.absolute_max:
                        self.absolute_max = spec_tfidf
            self.max_dict[doc_id] = max
            
    def tfidf(self,doc_id,word):
        idf = math.log(self.corpus_length / (1 + self.doc_freq_dict[word]))
        tf = self.tot_freq_dict[doc_id][word]
        return tf*idf

    def large_doc_normalized_tfidf(self,doc_id,word,a=0.4):
        idf = math.log(self.corpus_length / (1 + self.doc_freq_dict[word]))
        tf = self.tot_freq_dict[doc_id][word]
        full = tf*idf
        normalized = a + (1-a)*full/self.max_dict[doc_id]
        return normalized

    def small_doc_normalized_tfidf(self,doc_id,word,a=0.4):
        idf = math.log(self.corpus_length / (1 + self.doc_freq_dict[word]))
        tf = self.tot_freq_dict[doc_id][word]
        return tf*idf/self.absolute_max

    def top_n(self, doc_id, n=5,large_doc_normalized=False):
        if large_doc_normalized and small_doc_normalized:
            raise Exception("Must choose one or neither normalization type.")
        elif large_doc_normalized:
            scores = {word: self.large_doc_normalized_tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        elif small_doc_normalized:
            scores = {word: self.small_doc_normalized_tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        else:
            scores = {word: self.tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = {}
        for item in sorted_words[:n]:
            sorted_dict[item[0]] = item[1]
        return sorted_dict

    def every_word(self,doc_id,large_doc_normalized=False,small_doc_normalized=False):
        if large_doc_normalized and small_doc_normalized:
            raise Exception("Must choose one or neither normalization type.")
        elif large_doc_normalized:
            return {word: self.large_doc_normalized_tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        elif small_doc_normalized:
            return {word: self.small_doc_normalized_tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        else:
            return {word: self.tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}

    def save(self,directory_fp):
        if not os.path.exists(directory_fp):
            os.mkdir(directory_fp)

        with open(directory_fp + "/tot_freq.json", 'w') as tot_f:
            json.dump(self.tot_freq_dict,tot_f)
        with open(directory_fp + "/doc_freq.json", 'w') as doc_f:
            json.dump(self.doc_freq_dict,doc_f)
        with open(directory_fp + "/corpus.json", 'w') as corpus_f:
            json.dump(self.corpus_length,corpus_f)
    
    def load(self,directory_fp):
        with open(directory_fp + "/tot_freq.json", 'r') as tot_f:
            self.tot_freq_dict = json.load(tot_f)
        with open(directory_fp + "/doc_freq.json", 'r') as doc_f:
            self.doc_freq_dict = json.load(doc_f)
        with open(directory_fp + "/corpus.json", 'r') as corpus_f:
            self.corpus_length = json.load(corpus_f)