from textblob import TextBlob as tb

class TFIDF():
    def __init__(self):
        self.tot_freq_dict = {}
        self.doc_freq_dict = {}
        self.corpus_length = 0

    def batch_train_w_lists(self,str_list,id_list):
        if len(str_list) != len(id_list):
            raise Exception("Batch Training Error: Mismatched list sizes.")

        for index, item in enumerate(str_list):
            mini_dict = {}
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
        self.corpus_length += len(str_list)

    def batch_train_w_dict(self,str_dict):
        for item in str_dict:
            mini_dict = {}
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
        self.corpus_length += len(str_list)
            
    def tfidf(self,doc_id,word):
        idf = math.log(self.corpus_length / 1 + self.doc_freq_dict[word])
        tf = self.tot_freq_dict[doc_id][word]
        return tf*idf

    def top_n(self, doc_id, n):
        scores = {word: tfidf(doc_id,word) for word in self.tot_freq_dict[doc_id]}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]