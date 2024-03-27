import joblib

#load vectorizer (its basically the specific function used for the model to convert text to data)
vectorizer = joblib.load("tfidf_vectorizer_bigDataSet.pkl")

#load model (the trained model)
model = joblib.load("logistic_regression_model_bigDataSet.pkl")


# Read test essay from file
with open("TestEssay.txt", 'r', encoding='utf-8') as file:
    test_Essay = file.read()

#vectorize test essay using same method as model used
test_Essay_tfidf = vectorizer.transform([test_Essay])

prediction = model.predict(test_Essay_tfidf)

print("A human for sure wrote this" if prediction == 0 else "An AI wrote this for sure")
