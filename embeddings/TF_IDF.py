from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.preprocessing import clean_text


def perform_tfidf(x_train, x_test):

    tfidf_vectorizer = TfidfVectorizer(
        min_df=1,  # min count for relevant vocabulary
        max_features=4000,  # maximum number of features
        strip_accents='unicode',  # replace all accented unicode char
        # by their corresponding  ASCII char
        analyzer='word',  # features made of words
        token_pattern=r'\w{1,}',  # tokenize only words of 4+ chars
        ngram_range=(1, 1),  # features made of a single tokens
        use_idf=True,  # enable inverse-document-frequency reweighting
        smooth_idf=True,  # prevents zero division for unseen words
        sublinear_tf=False,
        preprocessor=clean_text)

    data_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    data_test_tfidf = tfidf_vectorizer.transform(x_test)

    return data_train_tfidf, data_test_tfidf
