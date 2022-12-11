# This implementation on car.csv dataset gives the same accuracy as Sklearn implementation of Naive Bayes algorithm
# By default Laplace Smoothing is 1
# import NaiveBayes class from naive_bayes.py
from naive_bayes import NaiveBayes

# get the data from dataset
data = pd.read_csv("car.csv", dtype="category", header = None)
data.columns = ["buying", "maint", "doors", "persons", "lug-boot", "safety", "accept"]

# split the dataset to training and testing data
X_train, y_train, X_test, y_test = train_test_split(data.values,test_size=0.25,random_state=0)
model = NaiveBayes(smoothing=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy(predictions, y_test)) # the accuracy must be 88.19444444444444, the same as Sklearn`s implementation

# without smoothing
# model = NaiveBayes(smoothing=False)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# accuracy(predictions, y_test)
