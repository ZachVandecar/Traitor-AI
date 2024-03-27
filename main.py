import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib


# Load the dataset
data = pd.read_csv("bigDataSet.csv")                                   #

# Split data into text (X) and labels (y)
X = data['text']    
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#it spits out the x used to train and the separate x used to test. same for y
#bascially, you want part of your set reserved for testing...the testing data shouldn't be used to train

#converts "frozen set" (whatever that is) to a list
stop_words_list = list(ENGLISH_STOP_WORDS)

# Convert text data to TF-IDF features with stop words removal
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words_list)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#save the model (probably works)  (saved, but specifically only works with the vectorizer function that I used before.
joblib.dump(classifier, "logistic_regression_model_bigDataSet.pkl")                    #

#save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer_bigDataSet.pkl")                        #