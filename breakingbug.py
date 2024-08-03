import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer  # This line enables IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv("dataset.csv")

# Exploring the data type of each column
df.info()

# Checking the data shape
df.shape

# Id column
df['id'].min(), df['id'].max()

# age column
df['age'].min(), df['age'].max()

# Let's summarize the age column
df['age'].describe()

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]

# Plot the histogram with custom colors
sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)

# Plot the mean, median, and mode of the age column using sns
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color='Green')
plt.axvline(df['age'].mode()[0], color='Blue')

# Print the value of mean, median, and mode of age column
print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())

# Plot the histogram of age column using plotly and coloring this by sex
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

# Find the values of the sex column
df['sex'].value_counts()

# Calculating the percentage of male and female value counts in the data
male_count = 726
female_count = 194

total_count = male_count + female_count

# Calculate percentages
male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

# Display the results
print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentage:.2f}%')

# Difference
difference_percentage = ((male_count - female_count) / female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

726 / 194

# Find the values count of age column grouping by sex column
df.groupby('sex')['age'].value_counts()

# Fix: Correct the column name to 'dataset'
df['dataset'].value_counts()

# Plot the countplot of dataset column
fig = px.bar(df, x='dataset', color='sex')
fig.show()

# Print the values of the dataset column grouped by sex
print(df.groupby('sex')['dataset'].value_counts())

# Make a plot of age column using plotly and coloring by dataset
fig = px.histogram(data_frame=df, x='age', color='dataset')
fig.show()

# Error Fix: Correct the syntax for printing mean, median, and mode of age column
print("___________________________________________________________")
print("Mean of the dataset: ", df['age'].mean())
print("___________________________________________________________")
print("Median of the dataset: ", df['age'].median())
print("___________________________________________________________")
print("Mode of the dataset: ", df['age'].mode()[0])
print("___________________________________________________________")

# Value count of cp column
df['cp'].value_counts()

# Count plot of cp column by sex column
sns.countplot(x='cp', hue='sex', data=df)

# Count plot of cp column by dataset column
sns.countplot(x='cp', hue='dataset', data=df)

# Draw the plot of age column grouped by cp column
fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

# Let's summarize the trestbps column
df['trestbps'].describe()

# Dealing with missing values in trestbps column
# Find the percentage of missing values in the trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() / len(df) * 100:.2f}%")

# Impute the missing values of the trestbps column using IterativeImputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
imputer1.fit(df[['trestbps']])

# Transform the data
df['trestbps'] = imputer1.transform(df[['trestbps']])

# Check the missing values in the trestbps column
print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

# First let's see data types or category of columns
df.info()

# Let's see which columns have missing values
(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

# Create an object of IterativeImputer
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# Columns that need to be imputed
imputed_columns = ['ca', 'oldpeak', 'chol', 'thalch']

# Fit the imputer on these columns and transform them
df[imputed_columns] = imputer2.fit_transform(df[imputed_columns])

# Let's check again for missing values
(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

# Error Fix: Correct the syntax for finding missing values
df.isnull().sum().sort_values(ascending=True)

missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
missing_data_cols

# Find categorical Columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

# Find Numerical Columns
num_cols = df.select_dtypes(exclude='object').columns.tolist()
num_cols

print(f'Categorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')

# Find columns
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg', 'thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak', 'age', 'restecg', 'fbs', 'cp', 'sex', 'num']

# For categorical data
def impute_categorical_missing_data(passed_col):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Encode categorical columns
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le  # Save the label encoder for later use

    # Encode the target variable if it's categorical
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    # Fit the RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    print(f"The feature '{passed_col}' has been imputed with {round(acc_score * 100, 2)}% accuracy\n")

    # Impute the missing values
    X_null = df_null.drop(passed_col, axis=1)
    
    # Encode the categorical columns in X_null using the previously fitted encoders
    X_null_encoded = X_null.copy()
    for col in X_null_encoded.select_dtypes(include=['object']).columns:
        le = label_encoders[col]
        # Handle previously unseen labels
        unseen_labels = set(X_null_encoded[col].unique()) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.concatenate([le.classes_, list(unseen_labels)])
        X_null_encoded[col] = le.transform(X_null_encoded[col])

    imputed_values = rf_classifier.predict(X_null_encoded)
    df.loc[df[passed_col].isnull(), passed_col] = y_le.inverse_transform(imputed_values)

    return df[passed_col]

# For numerical data
def impute_continuous_missing_data(passed_col):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Encode categorical columns using the entire dataset
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le  # Save the label encoder for later use

    # Fit the RandomForestRegressor
    rf_regressor = RandomForestRegressor(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    # Impute the missing values
    X_null = df_null.drop(passed_col, axis=1)
    
    # Encode the categorical columns in X_null using the previously fitted encoders
    X_null_encoded = X_null.copy()
    for col in X_null_encoded.select_dtypes(include=['object']).columns:
        le = label_encoders[col]
        X_null_encoded[col] = le.transform(X_null_encoded[col])

    imputed_values = rf_regressor.predict(X_null_encoded)
    df.loc[df[passed_col].isnull(), passed_col] = imputed_values

    return df[passed_col]

# Impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2)) + "%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(col)

df.isnull().sum().sort_values(ascending=False)

# Integrated debugged subplot logic
numerical_cols = ['age', 'trestbps', 'chol', 'thalch']

# Define a custom palette
palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]

plt.figure(figsize=(14, 8))

for i, col in enumerate(numerical_cols):
    plt.subplot(2, len(numerical_cols)//2, i+1)  # Adjust the number of rows and columns based on the number of plots
    sns.boxenplot(x=df[col], color=palette[i % len(palette)])  # Fix: Correct subplot and palette usage
    plt.title(col)
    plt.xlabel(col, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

plt.tight_layout()  # This will adjust subplot params to give specified padding.
plt.show()

# Additional plotting with modified settings
sns.set(rc={"axes.facecolor": "#B76E79", "figure.facecolor": "#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]

plt.figure(figsize=(14, 8))

for i, col in enumerate(numerical_cols):
    plt.subplot(2, len(numerical_cols)//2, i+1)  # Correct subplot settings
    sns.boxenplot(x=df[col], color=modified_palette[i % len(modified_palette)])  # Fix: Ensure correct subplot and color palette usage
    plt.title(col)
    plt.xlabel(col, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

# Night vision palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

plt.figure(figsize=(14, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, len(numerical_cols)//2, i+1)  # Correct subplot settings
    sns.boxenplot(x=df[col], color=night_vision_palette[i % len(night_vision_palette)])
    plt.title(col)
    plt.xlabel(col, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

# Splitting and encoding
X = df.drop('num', axis=1)
y = df['num']

Label_Encoder = LabelEncoder()
for col in categorical_cols:
    if df[col].dtype == 'object':
        df[col] = Label_Encoder.fit_transform(df[col].astype(str))

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define categorical columns for encoding
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Use ColumnTransformer to apply OneHotEncoder to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ], remainder='passthrough')

# Custom function to convert to dense only if the input is sparse
def convert_to_dense(X):
    if issparse(X):
        return X.toarray()
    return X

# Define the columns from the dataset
dataset_columns = [
    'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

# Create a list of models to evaluate
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', FunctionTransformer(convert_to_dense)),
        ('model', GaussianNB())
    ]))
]

best_model = None
best_accuracy = 0.0

# Iterate over the models and evaluate their performance
for name, model in models:
    if name == 'Naive Bayes Classifier':
        pipeline = model
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print the performance metrics
    print(f"| {name:<22} | {mean_accuracy:.4f} | {accuracy:.4f} |")

    # Check if the current model has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# Print the best model and its accuracy
print(f"\nBest Model: {type(best_model.named_steps['model']).__name__}")
print(f"Best Model Cross-Validation Accuracy: {mean_accuracy:.4f}")
print(f"Best Model Test Accuracy: {best_accuracy:.4f}")
