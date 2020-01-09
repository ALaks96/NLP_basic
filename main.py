import pandas as pd
from preprocessing.preprocessing import launch_preprocessing
from preprocessing.preprocessing import train_test
from preprocessing.preprocessing import concat_str_list
from embeddings.TF_IDF import perform_tfidf
from modelling.MultinomialNB_CLF import predict
from modelling.MultinomialNB_CLF import assess


train_model = True
path = 'data/spam_data_full.csv'

df = pd.read_csv(path)

# Preprocessing
df["clean_tokens"] = df["sms"].apply(launch_preprocessing)
df["clean_text"] = df["clean_tokens"].apply(concat_str_list)

# Training
x_train, x_test, y_train, y_test = train_test(df, "clean_text", "class")
x_train, x_test = perform_tfidf(x_train, x_test)

# Predicting
results = predict(x_train, y_train, x_test)
assess(results, y_test)



