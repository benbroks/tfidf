from textblob import TextBlob as tb
import pickle
import re

class TFIDF():
    def __init__(self,load_fp=None,clean=False):
        if load_fp is None:
            self.tot_freq_dict = {}
            self.doc_freq_dict = {}
            self.corpus_length = 0
            self.clean = clean
        else:
            self.load(load_fp)

    def preprocess(self,str_):
        str_ = str_.strip()
        str_ = str_.lower()
        str_ = re.sub('&',' and ',str_)
        str_ = re.sub(r'[^\w\d\s]+', ' ', str_)
        return str_

    def batch_train_w_lists(self,str_list,id_list,verbose=False):
        if len(str_list) != len(id_list):
            raise Exception("Batch Training Error: Mismatched list sizes.")

        # For Verbose
        num_docs = len(str_list)
        percent_cut = 0.05

        for index, item in enumerate(str_list):
            mini_dict = {}
            if self.clean:
                item = self.preprocess(item)
            blob = tb(item)
            doc_length = len(blob.words)
            for word in blob.words:
                if word not in mini_dict:
                    mini_dict[word] = blob.words.count(word) / doc_length
                    if word not in self.doc_freq_dict:
                        self.doc_freq_dict[word] = 1
                    else:
                        self.doc_freq_dict[word] += 1
            self.tot_freq_dict[id_list[index]] = mini_dict

            if verbose and ((i/num_docs) >= percent_cut):
                Print(i/num_docs*100, "percent complete.")
                percent_cut += 0.05

        self.corpus_length += len(str_list)

    def batch_train_w_dict(self,str_dict,verbose=False):
        # For Verbose
        num_docs = len(str_dict)
        percent_cut = 0.05

        for item in str_dict:
            mini_dict = {}
            if self.clean:
                str_dict[item] = self.preprocess(str_dict[item])
            blob = tb(str_dict[item])
            doc_length = len(blob.words)
            for word in blob.words:
                if word not in mini_dict:
                    mini_dict[word] = blob.words.count(word) / doc_length
                    if word not in self.doc_freq_dict:
                        self.doc_freq_dict[word] = 1
                    else:
                        self.doc_freq_dict[word] += 1
            self.tot_freq_dict[item] = mini_dict

            if verbose and ((i/num_docs) >= percent_cut):
                Print(i/num_docs*100, "percent complete.")
                percent_cut += 0.05

        self.corpus_length += len(str_list)
            
    def tfidf(self,doc_id,word):
        idf = math.log(self.corpus_length / 1 + self.doc_freq_dict[word])
        tf = self.tot_freq_dict[doc_id][word]
        return tf*idf

    def top_n(self, doc_id, n):
        scores = {word: tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]

    def save(self,fp):
        with open(fp, 'wb') as pickle_fp:
            pickle.dump(self,pickle_fp)
    
    def load(self,fp):
        with open(fp, 'rb') as pickle_fp:
            self = pickle.load(pickle_fp)