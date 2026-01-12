# Chuẩn bị data Churn
df['Churn'] = (df['Subscription Status'] == 'No').astype(int)  # 1: churn, 0: not

features_clf = [col for col in df.columns if col not in ['Churn', 'Subscription Status', 'Subscription Status_encoded', 'Cluster'] and 'encoded' in col or col in numerical_cols]
X_clf = df[features_clf]
y_clf = df['Churn']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_clf, y_train_clf)
y_pred_log = logreg.predict(X_test_clf)
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print(classification_report(y_test_clf, y_pred_log))

# Model 2: Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf_clf = rf_clf.predict(X_test_clf)
print("Random Forest Accuracy:", accuracy_score(y_test_clf, y_pred_rf_clf))
print(classification_report(y_test_clf, y_pred_rf_clf))
