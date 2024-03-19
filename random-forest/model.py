# Load useful libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from dataloader import load_data

# Create Train & Test Data
X_train, y_train, X_test, y_test = load_data()

# Build the model
rf_clf = RandomForestClassifier(max_features=3, n_estimators =100 ,bootstrap = True)
rf_clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf_clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))