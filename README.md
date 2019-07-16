# tfidf

Implementation of Term Frequency - Inverse Document Frequency using [TextBlob](https://textblob.readthedocs.io/en/dev/index.html). [This algorithm](http://www.tfidf.com/) is typically used to measure the relative importance of words in documents without relying exclusively on raw frequency. Raw frequency disproportionately values stop words and other common words that show up in nearly any corpus.

From what I've seen, implementations of tf-idf rarely allow for batch training. Given that I'm often working with datasets of large-ish sizes (~100k docs), I decided to build one out. 

Data can be input in a couple different ways:
1. List of Strings + Corresponding List of String IDs. (i.e. String ID at index i corresponds to the String at index i)
    - call `batch_train_w_lists`
2. Dictionary of ID keys + String values. Each ID directly maps to a String.
    - call `batch_train_w_dict`

When initializing your TFIDF object, toggle the boolean `clean` parameter to apply string pre-processing prior to training. This way, words like "PaNcakes" and "pancakes" will be considered one and the same! One thing to note: symbols will be replaced with spaces, so be prepared for "Ben's" to be converted to "ben" and "s", two separate words.

Output structures:
1. tfidf(doc_id,word): Returns typical tf-idf value
2. large_doc_normalized_tfidf(doc_id,word): Largest tf-idf value in a given document is 1. Inspired by [Stanford's description](https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html).
    - First, we calculate the typical value, _t_ = tfidf(doc_id,word)
    - Second, we calculate the maximum tf-idf value within _doc-id_, _max_.
    - Choose some _a_ (usually 0.4), return _a_ + (1-_a_)*_t_/_max_.
3. small_doc_normalized_tfidf(doc_id,word): Largest tf-idf value in the entire corpus is 1.
    - After calculating every typical tf-idf value, we find the _absolute max_.
    - Return tfidf(doc_id,word)/_absolute max_.

Larger Scale Outputs: 
1. top_n(doc_id,n=5,large_doc_normalized=False,small_doc_normalized=False): Returns, in order by tfidf value, the n greatest words in a document.
2. every_word(doc_id,large_doc_normalized=False,small_doc_normalized=False): Returns every word and corresponding tfidf value in a given document.
