
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
# Load data
df = pd.read_csv("student_selection_data.csv")

# Features and target
X = df.drop("selected", axis=1)
y = df["selected"]

# Preprocessing
numeric_features = ['aptitude_score', 'programming_score', 'communication_score', 'gpa', 'project_score']
#categorical_features = ['english_fluency']

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

#categorical_transformer = Pipeline(steps=[
    #("imputer", SimpleImputer(strategy="most_frequent")),
    #("onehot", OneHotEncoder(handle_unknown="ignore"))
#])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
    #("cat", categorical_transformer, categorical_features)
])

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Predict & Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
with open("evaluation.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
with open('interview_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
