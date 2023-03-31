import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading the csv file (requires the part 1 for news_file to exist)
df = pd.read_csv("news_file.csv")

# spliting the datas into training (80%), validation (20%) sets
train_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)

# converting the content to a matrix of word frequency counts
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['content'])
X_validation = vectorizer.transform(validation_data['content'])

y_train = train_data['label']
y_validation = validation_data['label']

# training the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# making some predictions and calculating the accuracy
y_pred = model.predict(X_validation)
accuracy = accuracy_score(y_validation, y_pred)

print("accuracy:", round(accuracy, 2))
