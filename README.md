# tfidf

Implementation of Term Frequency - Inverse Document Frequency using TextBlob. This algorithm is typically used to measure the relative importance of words in documents without relying exclusively on raw frequency. Raw frequency disproportionately values stop words and other common words that show up in nearly any corpus.

From what I've seen, implementations of tf-idf rarely allow for batch training. Given that I'm often working with datasets of large-ish sizes (~100k docs), I decided to build one out. 

Data can be input in a couple different ways:
1. List of Strings + Corresponding List of String IDs. (i.e. String ID at index i corresponds to the String at index i)
    - call `batch_train_w_lists`
2. Dictionary of ID keys + String values. Each ID directly maps to a String.
    - call `batch_train_w_dict`

When initializing your TFIDF object, toggle the boolean `clean` parameter to apply string pre-processing prior to training. This way, words like "PaNcakes" and "pancakes" will be considered one and the same! One thing to note: symbols will be replaced with space, so be prepared for "Ben's" to be converted to "ben" and "s", two separate words.