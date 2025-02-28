#!/usr/bin/env python
# coding: utf-8

# #  Business Understanding
#
#
#
#

# In the telecom industry, customer churn remains one of the most significant challenges due to intense competition and evolving customer expectations. Churn not only leads to revenue loss but also increases the cost of acquiring new customers, which is far higher than retaining existing ones. Predicting churn and understanding the underlying causes are essential for implementing proactive retention strategies and enhancing customer satisfaction.
#
# This machine learning project focuses on reproducing and expanding upon findings from key research papers that explore customer churn prediction using data-driven methods. By analyzing a telecom customer churn dataset—consisting of attributes like account duration, international plan subscription, call usage, service charges, customer service interactions, and churn status—the project aims to achieve the following:
# *   Build effective churn prediction models to identify at-risk customers.
# *   Analyze key attributes influencing churn behavior (e.g., service plans, call patterns, or customer support usage).
# *   Offer actionable insights to optimize customer retention efforts and reduce churn rates.
#
# The outcome of this analysis will empower telecom providers to anticipate churn early, focus on high-risk customers, and tailor their strategies to improve customer loyalty and business performance.
#
#

# In[105]:


# =============================
# 1. Essential Libraries
# =============================
import numpy as np
import pandas as pd
import warnings

# =============================
# 2. Data Visualization
# =============================
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Interactive Visualization
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px

# =============================
# 3. Preprocessing
# =============================
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler

# =============================
# 4. Feature Selection
# =============================
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    SelectFromModel,
)

# =============================
# 5. Model Training & Evaluation
# =============================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensemble Models
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)

# Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier

# =============================
# 6. Model Selection & Validation
# =============================
from sklearn.model_selection import train_test_split, GridSearchCV

# =============================
# 7. Model Evaluation Metrics
# =============================
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)

# =============================
# 8. Other Utilities
# =============================
from math import ceil
import itertools
import pickle

# =============================
# 9. General Settings
# =============================
warnings.filterwarnings("ignore")  # Suppress all warnings
sns.set_theme(style="whitegrid")  # Set default seaborn style
plt.style.use("seaborn-v0_8-pastel")  # Set matplotlib style


# In[5]:


# Load the datasets

df_80 = pd.read_csv("/content/sample_data/churn-bigml-80.csv")
df_20 = pd.read_csv("/content/sample_data/churn-bigml-20.csv")


# # Data Understanding

# ## 1) Data Overview

# In[6]:


# Display basic information about the datasets
def display_basic_info(df, name):
    print(f"\nDataset: {name}")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())


# Display information for both datasets
display_basic_info(df_20, "churn-bigml-20")
display_basic_info(df_80, "churn-bigml-80")


# *   State: The US state in which the customer resides.
# *   Account length:The number of days the customer has had an active account with the telecom service provider.
# *   Area code: The telephone area code assigned to the customer.
# *   International plan: Whether the customer is subscribed to an international calling plan (Yes/No).
# *   Voice mail plan: Whether the customer has subscribed to a voicemail plan (Yes/No).
# *   Number vmail messages: The number of voicemail messages the customer has received.
# *   Total day minutes: The total minutes used by the customer during the day for calls.
# *   Total day calls: The total number of calls made by the customer during the day.
# *   Total day charge: The total charge incurred by the customer for daytime calls.
# *   Total eve minutes: The total minutes used by the customer during the evening for calls.
# *   Total eve calls: The total number of calls made by the customer during the evening.
# *   Total eve charge: The total charge incurred by the customer for evening calls.
# *   Total night minutes: The total minutes used by the customer during the night for calls.
# *   Total night calls: The total number of calls made by the customer during the night.
# *   Total night charge: The total charge incurred by the customer for nighttime calls.
# *   Total intl minutes: The total minutes used by the customer for international calls.
# *   Total intl calls: The total number of international calls made by the customer.
# *   Total intl charge: The total charge incurred by the customer for international calls.
# *   Customer service calls: The total number of calls the customer has made to customer service.
# *   Churn: Whether the customer discontinued their service with the company (Yes/No).

# ## 2) Exploratory data Analysis

#  data distribution

# In[7]:


# Identify numerical and categorical columns
numerical_columns = df_80.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df_80.select_dtypes(include=[object, "bool"]).columns.tolist()

# Plot distribution of numerical features
plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(5, 4, i)
    sns.histplot(df_80[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# Plot distribution of categorical features
plt.figure(figsize=(10, 10))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(5, 4, i)
    sns.countplot(x=df_80[column])
    plt.title(column)
plt.tight_layout()
plt.show()


# The histograms for the numerical columns in the churn-bigml-80 dataset reveal that most of the columns do not follow a normal distribution
#

# In[8]:


# Visualization of pie charts regarding service status targets for Churn customers
# View classes from Telco Customer Churn
df_80["Churn"].value_counts().plot(kind="pie", figsize=(5, 5), autopct="%.2f")
sns.set_theme(style="darkgrid")


# In[9]:


# Visualization of pie charts regarding service status targets for Churn customers
# View classes from Telco Customer Churn
df_20["Churn"].value_counts().plot(kind="pie", figsize=(5, 5), autopct="%.2f")
sns.set_theme(style="darkgrid")


# # Data Preparation

# 1) data cleaning

# missing values:

# In[10]:


# Calculate the percentage of missing values in each column
missing_percentage = df_80.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]
missing_percentage.sort_values(ascending=False)


# In[11]:


missing_percentage_20 = df_20.isnull().mean() * 100
missing_percentage_20 = missing_percentage_20[missing_percentage_20 > 0]
missing_percentage_20.sort_values(ascending=False)


# handling outliers

# In[12]:


# Function to create box plots for outliers
import plotly.express as px
import plotly.graph_objects as go


def plot_outliers(df, title):
    fig = go.Figure()
    for column in df.select_dtypes(include=["float64", "int64"]).columns:
        fig.add_trace(go.Box(y=df[column], name=column))
    fig.update_layout(title=title, yaxis_title="Values")
    return fig


# Plot outliers for df_80 and df_20
fig_80 = plot_outliers(df_80, "Outliers in churn-bigml-80")
fig_20 = plot_outliers(df_20, "Outliers in churn-bigml-20")

fig_80.show()
fig_20.show()


# we chose to i
# mpute the outliers with the median because the median is a robust measure of central tendency that is not influenced by extreme values.

# In[13]:


# Function to impute outliers using the IQR method
def impute_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_value = df[column].median()
        df[column] = np.where(
            (df[column] < lower_bound) | (df[column] > upper_bound),
            median_value,
            df[column],
        )
    return df


# Numerical columns in the dataset
numerical_columns = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

# Impute outliers in the numerical columns
df_80_imputed = impute_outliers(df_80.copy(), numerical_columns)
df_20_imputed = impute_outliers(df_20.copy(), numerical_columns)

df_80_imputed.head()


# In[14]:


# Impute outliers in the numerical columns
df_80_imputed = impute_outliers(df_80.copy(), numerical_columns)
df_20_imputed = impute_outliers(df_20.copy(), numerical_columns)

# Plot outliers for df_80_imputed and df_20_imputed
fig_80_imputed = plot_outliers(
    df_80_imputed, "Outliers in churn-bigml-80 after Imputation"
)
fig_20_imputed = plot_outliers(
    df_20_imputed, "Outliers in churn-bigml-20 after Imputation"
)

fig_80_imputed.show()
fig_20_imputed.show()


# In[15]:


# Identify categorical features and their types (ordered or unordered)
categorical_features = df_80.select_dtypes(include=["object"]).columns

# Determine if categorical features are ordered or unordered
categorical_types = {}
for feature in categorical_features:
    unique_values = df_80[feature].unique()
    if pd.api.types.is_categorical_dtype(df_80[feature]) and df_80[feature].cat.ordered:
        categorical_types[feature] = "Ordered"
    else:
        categorical_types[feature] = "Unordered"

categorical_types


# 2) data transformation

# data encoding

# In[16]:


# Identify categorical features
categorical_features = ["State", "International plan", "Voice mail plan"]

# Initialize the OrdinalEncoder
encoder = OrdinalEncoder()

# Fit and transform the categorical features for df_80_imputed
encoded_80 = encoder.fit_transform(df_80_imputed[categorical_features])
encoded_df_80 = df_80_imputed.copy()
encoded_df_80[categorical_features] = encoded_80

# Transform the categorical features for df_20_imputed
encoded_20 = encoder.transform(df_20_imputed[categorical_features])
encoded_df_20 = df_20_imputed.copy()
encoded_df_20[categorical_features] = encoded_20

# Encode the Churn feature
encoded_df_80["Churn"] = encoded_df_80["Churn"].astype(int)
encoded_df_20["Churn"] = encoded_df_20["Churn"].astype(int)

# Keep the original order of the columns
encoded_df_80 = encoded_df_80[df_80_imputed.columns]
encoded_df_20 = encoded_df_20[df_20_imputed.columns]

encoded_df_80.head()
encoded_df_20.head()


# feautre scaling : Normalization

# In[17]:


# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical features for encoded_df_80
scaled_80 = scaler.fit_transform(encoded_df_80)
scaled_df_80 = pd.DataFrame(scaled_80, columns=encoded_df_80.columns)

# Transform the numerical features for encoded_df_20
scaled_20 = scaler.transform(encoded_df_20)
scaled_df_20 = pd.DataFrame(scaled_20, columns=encoded_df_20.columns)

scaled_df_80.head()
scaled_df_20.head()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Numerical columns in the dataset
numerical_columns = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

# Plot histograms for the numerical columns after scaling
plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(4, 5, i)
    sns.histplot(scaled_df_80[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(4, 5, i)
    sns.histplot(scaled_df_20[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()


# Feature selection

# In[19]:


# Select only numerical columns for correlation matrix
numerical_columns = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

# Compute the correlation matrix for the 80% dataset
correlation_matrix = scaled_df_80[numerical_columns].corr()

# Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix for 80% Dataset")
plt.show()


# In[20]:


# Remove one feature from each highly correlated pair
features_to_remove = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

# Create a new dataframe with the selected features removed
reduced_df_80 = scaled_df_80.drop(columns=features_to_remove, axis=1)

# Display the first few rows of the reduced dataframe
reduced_df_80.head()


# In[21]:


# Remove one feature from each highly correlated pair
features_to_remove = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

# Create a new dataframe with the selected features removed
reduced_df_20 = scaled_df_20.drop(columns=features_to_remove)

# Display the first few rows of the reduced dataframe
reduced_df_20.head()


# ##modeling

# ##The Logistic Regression model

# In[22]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]}

# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation for the reduced dataset
grid_search_reduced = GridSearchCV(
    estimator=log_reg, param_grid=param_grid, cv=5, scoring="accuracy"
)

# Fit the model to the reduced data
grid_search_reduced.fit(X_reduced, y_reduced)

# Get the best parameters and the best score for the reduced dataset
best_params_reduced = grid_search_reduced.best_params_
best_score_reduced = grid_search_reduced.best_score_

best_params_reduced, best_score_reduced


# The Logistic Regression model was optimized using hyperparameter tuning and 5-fold cross-validation on the reduced_df_80 dataset. The best parameters identified were a regularization parameter ( C = 1 ) and the liblinear solver. This configuration achieved an average cross-validation accuracy of approximately 86.35%, indicating a well-performing model for predicting customer churn in this reduced dataset.

# ##The K-Nearest Neighbors (KNN) model

# In[23]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}

# Initialize the KNN model
knn = KNeighborsClassifier()

# Initialize GridSearchCV with 5-fold cross-validation for the reduced dataset
grid_search_knn = GridSearchCV(
    estimator=knn, param_grid=param_grid_knn, cv=5, scoring="accuracy"
)

# Fit the model to the reduced data
grid_search_knn.fit(X_reduced, y_reduced)

# Get the best parameters and the best score for the reduced dataset
best_params_knn = grid_search_knn.best_params_
best_score_knn = grid_search_knn.best_score_

best_params_knn, best_score_knn


# ##Support Vector Machine (SVM)

# In[24]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"],
}

# Initialize the SVM model
svm = SVC()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=svm, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

# Fit the model to the data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best SVM model on the entire dataset
best_svm = grid_search.best_estimator_
best_svm.fit(X_reduced, y_reduced)

# Print the best parameters and best score
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")


# ##decision tree

# In[25]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "max_depth": [None, 1, 5, 4, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
}

# Initialize the Decision Tree model
dt = DecisionTreeClassifier()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

# Fit the model to the data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best Decision Tree model on the entire dataset
best_dt = grid_search.best_estimator_
best_dt.fit(X_reduced, y_reduced)

# Print the best parameters and best score
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.4f}")


# In[26]:


from sklearn.tree import export_graphviz
from io import StringIO
import graphviz
from IPython.display import Image

# Create a StringIO object to hold the dot data
dot_data = StringIO()

# Export the tree to dot format
export_graphviz(
    best_dt,
    out_file=dot_data,
    filled=True,
    feature_names=X_reduced.columns,
    class_names=["No Churn", "Churn"],
    rounded=True,
    special_characters=True,
)

# Create a graph from the dot data
graph = graphviz.Source(dot_data.getvalue())

# Display the tree image
# Change format to 'png' to generate a PNG file
graph.render("decision_tree", format="png")
Image(filename="decision_tree.png")


# ##random forest

# In[27]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [2 * n + 1 for n in range(20)],  # Number of trees in the forest
    "max_depth": [2 * n + 1 for n in range(10)],  # Maximum depth of the tree
    "max_features": [
        "auto",
        "sqrt",
        "log2",
    ],  # Number of features to consider for the best split
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_rf = grid_search.best_estimator_
best_rf.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# ##AdaBoost

# In[28]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [
        0.1 * (n + 1) for n in range(10)
    ],  # Contribution of each weak classifier
    "n_estimators": [
        2 * n + 1 for n in range(10)
    ],  # Number of weak classifiers to use in the ensemble
    "algorithm": ["SAMME", "SAMME.R"],  # Type of boosting algorithm to use
}

# Initialize the AdaBoost classifier
ada = AdaBoostClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=ada, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_ada = grid_search.best_estimator_
best_ada.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# ##XGboost

# In[29]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [
        0.01,
        0.1,
        0.2,
        0.3,
    ],  # Step size shrinkage used to prevent overfitting
    "n_estimators": [10, 20, 50],  # Number of boosting rounds
    "max_depth": [3, 5, 7],  # Maximum depth of a tree
    "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    "colsample_bytree": [
        0.6,
        0.8,
        1.0,
    ],  # Subsample ratio of columns when constructing each tree
}

# Initialize the XGBoost classifier
xgb = XGBClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# ##GBM

# In[30]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [
        0.01,
        0.1,
        0.2,
        0.3,
    ],  # Step size shrinkage used to prevent overfitting
    "n_estimators": [10, 20, 50],  # Number of boosting rounds
    "max_depth": [3, 5, 7],  # Maximum depth of a tree
    "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    "max_features": [
        "auto",
        "sqrt",
        "log2",
    ],  # Number of features to consider for the best split
}

# Initialize the Gradient Boosting classifier
gbm = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=gbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_gbm = grid_search.best_estimator_
best_gbm.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# ##LGBM

# In[55]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [
        0.01,
        0.1,
        0.2,
        0.3,
    ],  # Step size shrinkage used to prevent overfitting
    "n_estimators": [10, 20, 50],  # Number of boosting rounds
    "max_depth": [3, 5, 7],  # Maximum depth of a tree
    "num_leaves": [31, 50, 100],  # Number of leaves in one tree
    "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    "colsample_bytree": [
        0.6,
        0.8,
        1.0,
    ],  # Subsample ratio of columns when constructing each tree
}

# Initialize the LightGBM classifier
lgbm = lgb.LGBMClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=lgbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_lgbm = grid_search.best_estimator_
best_lgbm.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# ##Bagging classifier model

# In[32]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced = reduced_df_80.drop(columns=["Churn"])
y_reduced = reduced_df_80["Churn"]


# Define the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [10, 50, 100, 200],  # Number of base estimators in the ensemble
    "max_samples": [
        0.5,
        0.7,
        1.0,
    ],  # The number of samples to draw from X to train each base estimator
    "max_features": [
        0.5,
        0.7,
        1.0,
    ],  # The number of features to draw from X to train each base estimator
    "bootstrap": [True, False],  # Whether samples are drawn with replacement
    "bootstrap_features": [True, False],  # Whether features are drawn with replacement
}

# Initialize the Bagging classifier with a DecisionTreeClassifier as the base estimator
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=bagging, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
)

# Fit the model to the training data
grid_search.fit(X_reduced, y_reduced)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the best model on the entire training set
best_bagging = grid_search.best_estimator_
best_bagging.fit(X_reduced, y_reduced)

# Print the best parameters and the best score
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)


# # Evalution

# ##Logistic Regression classifie

# In[46]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the Logistic Regression classifier with the best parameters
logreg_model = LogisticRegression(C=1, solver="lbfgs", max_iter=1000)

# Fit the model to the training data
logreg_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_logreg = logreg_model.predict(X_reduced_test)

# Evaluate the model
accuracy_logreg = accuracy_score(y_reduced_test, y_pred_logreg)
report_logreg = classification_report(y_reduced_test, y_pred_logreg)

# Print the accuracy and classification report
print("Accuracy:", accuracy_logreg)
print("Classification Report:\n", report_logreg)


# ##the KNN classifier

# In[43]:


# Define the feature matrix and target vector for reduced_df_80
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]
# Define the feature matrix and target vector for reduced_df_80
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]
# Best parameters found
best_params = {"metric": "manhattan", "n_neighbors": 7, "weights": "distance"}
# Initialize the KNN classifier with the best parameters
knn = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    metric=best_params["metric"],
    weights=best_params["weights"],
)
# Fit the model to the training data
knn_model = knn.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_knn = knn_model.predict(X_reduced_test)

# Evaluate the model
accuracy = accuracy_score(y_reduced_test, y_pred_knn)
report = classification_report(y_reduced_test, y_pred_knn)

# Print the accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# ##Support Vector Machine (SVM)

# In[47]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the SVM classifier with the best parameters
svm_model = SVC(C=10, gamma="scale", kernel="rbf")

# Fit the model to the training data
svm_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_svm = svm_model.predict(X_reduced_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_reduced_test, y_pred_svm)
report_svm = classification_report(y_reduced_test, y_pred_svm)

# Print the accuracy and classification report
print("Accuracy:", accuracy_svm)
print("Classification Report:\n", report_svm)


# ##DecisionTree

# In[67]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the Decision Tree classifier with the best parameters
dt_model = DecisionTreeClassifier(
    criterion="gini", max_depth=5, min_samples_leaf=2, min_samples_split=5
)

# Fit the model to the training data
dt_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_dt = dt_model.predict(X_reduced_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_reduced_test, y_pred_dt)
report_dt = classification_report(y_reduced_test, y_pred_dt)

# Print the accuracy and classification report
print("Accuracy:", accuracy_dt)
print("Classification Report:\n", report_dt)


# ##RandomForest

# In[50]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the Random Forest classifier with the best parameters
rf_model = RandomForestClassifier(max_depth=13, max_features="sqrt", n_estimators=39)

# Fit the model to the training data
rf_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_rf = rf_model.predict(X_reduced_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_reduced_test, y_pred_rf)
report_rf = classification_report(y_reduced_test, y_pred_rf)

# Print the accuracy and classification report
print("Accuracy:", accuracy_rf)
print("Classification Report:\n", report_rf)


# ##AdaBoost

# In[51]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the AdaBoost classifier with the best parameters
ada_model = AdaBoostClassifier(algorithm="SAMME.R", learning_rate=1.0, n_estimators=13)

# Fit the model to the training data
ada_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_ada = ada_model.predict(X_reduced_test)

# Evaluate the model
accuracy_ada = accuracy_score(y_reduced_test, y_pred_ada)
report_ada = classification_report(y_reduced_test, y_pred_ada)

# Print the accuracy and classification report
print("Accuracy:", accuracy_ada)
print("Classification Report:\n", report_ada)


# ##XGBoost

# In[90]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the XGBoost classifier with the best parameters
xgb_model = xgb.XGBClassifier(
    colsample_bytree=1.0, learning_rate=0.1, max_depth=5, n_estimators=50, subsample=1.0
)

# Fit the model to the training data
xgb_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_xgb = xgb_model.predict(X_reduced_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_reduced_test, y_pred_xgb)
report_xgb = classification_report(y_reduced_test, y_pred_xgb)

# Print the accuracy and classification report
print("Accuracy:", accuracy_xgb)
print("Classification Report:\n", report_xgb)


# ##Gradient Boosting Machine (GBM)

# In[56]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the Gradient Boosting classifier with the best parameters
gbm_model = GradientBoostingClassifier(
    learning_rate=0.3, max_depth=7, max_features="sqrt", n_estimators=50, subsample=1.0
)

# Fit the model to the training data
gbm_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_gbm = gbm_model.predict(X_reduced_test)

# Evaluate the model
accuracy_gbm = accuracy_score(y_reduced_test, y_pred_gbm)
report_gbm = classification_report(y_reduced_test, y_pred_gbm)

# Print the accuracy and classification report
print("Accuracy:", accuracy_gbm)
print("Classification Report:\n", report_gbm)


# ##LightGBM (LGBM)

# In[57]:


# Define the feature matrix and target vector for the training set
# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the LightGBM classifier with the best parameters
lgbm_model = lgb.LGBMClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=50,
    num_leaves=50,
    subsample=0.6,
)

# Fit the model to the training data
lgbm_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_lgbm = lgbm_model.predict(X_reduced_test)

# Evaluate the model
accuracy_lgbm = accuracy_score(y_reduced_test, y_pred_lgbm)
report_lgbm = classification_report(y_reduced_test, y_pred_lgbm)

# Print the accuracy and classification report
print("Accuracy:", accuracy_lgbm)
print("Classification Report:\n", report_lgbm)


# ##the Bagging model
#

# In[64]:


# Define the feature matrix and target vector for the training set
X_reduced_train = reduced_df_80.drop(columns=["Churn"])
y_reduced_train = reduced_df_80["Churn"]

# Define the feature matrix and target vector for the testing set
X_reduced_test = reduced_df_20.drop(columns=["Churn"])
y_reduced_test = reduced_df_20["Churn"]

# Create the Bagging classifier with the best parameters
bagging_model = BaggingClassifier(
    bootstrap=False,
    bootstrap_features=False,
    max_features=1.0,
    max_samples=0.5,
    n_estimators=200,
)

# Fit the model to the training data
bagging_model.fit(X_reduced_train, y_reduced_train)

# Predict on the test data
y_pred_bagging = bagging_model.predict(X_reduced_test)

# Evaluate the model
accuracy_bagging = accuracy_score(y_reduced_test, y_pred_bagging)
report_bagging = classification_report(y_reduced_test, y_pred_bagging)

# Print the accuracy and classification report
print("Accuracy:", accuracy_bagging)
print("Classification Report:\n", report_bagging)


# ## Model performance metrics

# In[92]:


# Suppress warnings
warnings.filterwarnings("ignore")

# Load or prepare your data
X_train = reduced_df_80.drop(columns=["Churn"])
y_train = reduced_df_80["Churn"]
X_test = reduced_df_20.drop(columns=["Churn"])
y_test = reduced_df_20["Churn"]

# Define Models, including Gradient Boosting
models = {
    "Bagging Classifier": BaggingClassifier(
        bootstrap=False,
        bootstrap_features=False,
        max_features=1.0,
        max_samples=0.5,
        n_estimators=200,
    ),
    "KNN Classifier": KNeighborsClassifier(
        n_neighbors=7, metric="manhattan", weights="distance"
    ),
    "Logistic Regression": LogisticRegression(C=1, solver="lbfgs", max_iter=1000),
    "SVM (RBF Kernel)": SVC(C=10, gamma="scale", kernel="rbf", probability=True),
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=2, min_samples_split=5
    ),
    "Random Forest": RandomForestClassifier(
        max_depth=13, max_features="sqrt", n_estimators=39
    ),
    "AdaBoost": AdaBoostClassifier(
        algorithm="SAMME.R", learning_rate=1.0, n_estimators=13
    ),
    "XGBoost": xgb.XGBClassifier(
        colsample_bytree=1.0,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=50,
        subsample=1.0,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        learning_rate=0.3,
        max_depth=7,
        max_features="sqrt",
        n_estimators=50,
        subsample=1.0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=7,
        n_estimators=50,
        num_leaves=50,
        subsample=0.6,
    ),
}

# List to store model performance
model_results = []

# Evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    kappa = cohen_kappa_score(y_test, y_pred)

    # Append results
    model_results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1-score": f1,
            "ROC_AUC": roc_auc,
        }
    )

# Create a DataFrame
results_df = pd.DataFrame(model_results)

# Round the metrics for better presentation
results_df = results_df.round(4)

# Display table using Plotly
table = ff.create_table(results_df)
pio.show(table)

# Print classification reports for all models
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))


# Key Observations:
# - Bagging Classifier has the highest Accuracy (0.922) and high Precision (0.9216) but a moderate Recall (0.4947).
# - Decision Tree achieves a high Accuracy (0.9175) and the highest Recall (0.5368), indicating it recovers more true positives compared to others.
# - XGBoost also performs very well:
#  Accuracy = 0.919,
#  Recall = 0.4947,
#  Precision = 0.8868,
#  f1-score = 0.6351,
#  ROC_AUC = 0.8432.
# - Gradient Boosting and LightGBM have similar performance:
# Accuracy = 0.9145,
# ROC_AUC scores are close to XGBoost but slightly lower.
# - Final Recommendation:
# XGBoost is the best overall model:
#
# Accuracy and ROC_AUC are near the top.
# It balances Precision and Recall well, achieving a good f1-score of 0.6351.

#

# ##Compare model metrics

# In[86]:


def output_tracer(df, metric, color):
    tracer = go.Bar(
        y=df["Model"],
        x=df[metric],
        orientation="h",
        name=metric,
        marker=dict(line=dict(width=0.7), color=color),
    )
    return tracer


def modelmetricsplot(df, title):
    layout = go.Layout(
        dict(
            title=title,
            plot_bgcolor="rgb(243,243,243)",
            paper_bgcolor="rgb(243,243,243)",
            xaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                title="Metric",
                zerolinewidth=1,
                ticklen=5,
                gridwidth=2,
            ),
            yaxis=dict(
                gridcolor="rgb(255, 255, 255)", zerolinewidth=1, ticklen=5, gridwidth=2
            ),
            margin=dict(l=250),
            height=780,
        )
    )

    # Trace for each metric with pastel colors
    trace1 = output_tracer(df, "Accuracy", "#AEC6CF")  # Pastel blue
    trace2 = output_tracer(df, "Recall", "#FFB3BA")  # Pastel red
    trace3 = output_tracer(df, "Precision", "#FFB347")  # Pastel orange
    trace4 = output_tracer(df, "f1-score", "#FFDFD3")  # Pastel pink
    trace5 = output_tracer(df, "ROC_AUC", "#D1C4E9")  # Pastel purple

    # Combine traces into data for the plot
    data = [trace1, trace2, trace3, trace4, trace5]
    fig = go.Figure(data=data, layout=layout)

    # Display the plot
    fig.show()


# Call the plotting function with the DataFrame and title
modelmetricsplot(df=results_df, title="Model Performances over the Training Dataset")


# ##Confusion matrices for models

# In[94]:


warnings.filterwarnings("ignore")  # Suppress all warnings


# Load or prepare your data
X_train = reduced_df_80.drop(columns=["Churn"])
y_train = reduced_df_80["Churn"]
X_test = reduced_df_20.drop(columns=["Churn"])
y_test = reduced_df_20["Churn"]

# Define Models
models = {
    "Bagging Classifier": BaggingClassifier(
        bootstrap=False,
        bootstrap_features=False,
        max_features=1.0,
        max_samples=0.5,
        n_estimators=200,
    ),
    "KNN Classifier": KNeighborsClassifier(
        n_neighbors=7, metric="manhattan", weights="distance"
    ),
    "Logistic Regression": LogisticRegression(C=1, solver="lbfgs", max_iter=1000),
    "SVM (RBF Kernel)": SVC(C=10, gamma="scale", kernel="rbf", probability=True),
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=2, min_samples_split=5
    ),
    "Random Forest": RandomForestClassifier(
        max_depth=13, max_features="sqrt", n_estimators=39
    ),
    "AdaBoost": AdaBoostClassifier(
        algorithm="SAMME.R", learning_rate=1.0, n_estimators=13
    ),
    "XGBoost": xgb.XGBClassifier(
        colsample_bytree=1.0,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=50,
        subsample=1.0,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        learning_rate=0.3,
        max_depth=7,
        max_features="sqrt",
        n_estimators=50,
        subsample=1.0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=7,
        n_estimators=50,
        num_leaves=50,
        subsample=0.6,
    ),
}


# Plotting Confusion Matrix Function
def confmatplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber):
    fig = plt.figure(
        figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber))
    )
    fig.set_facecolor("#F3F3F3")
    pastel_cmap = sns.color_palette("pastel", as_cmap=True)  # Pastel colormap

    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)
        model = modeldict[name].fit(df_train, target_train)
        predictions = model.predict(df_test)
        conf_matrix = confusion_matrix(target_test, predictions)

        # Heatmap with pastel colors
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            square=True,
            xticklabels=["Not churn", "Churn"],
            yticklabels=["Not churn", "Churn"],
            linewidths=2,
            linecolor="w",
            cmap=pastel_cmap,
        )

        plt.title(name, color="b")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)


# Call the plotting function
confmatplot(
    modeldict=models,
    df_train=X_train,
    df_test=X_test,
    target_train=y_train,
    target_test=y_test,
    figcolnumber=3,
)


# XGBoost emerges as the best overall model:
#
#
#
#
#  - Balanced performance across FP, FN, and TP.
#  - Low misclassification rates and better generalization.
# Bagging Classifier and LightGBM are close competitors with slightly different trade-offs:
#
# Bagging Classifier:
# - Very low FP but slightly higher FN.
# LightGBM:
# - Balanced but slightly lower TP compared to XGBoost.
#
# For business-critical predictions :XGBoost provides the best balance.

# ##ROC - Curves for models

# In[81]:


def rocplot(modeldict, df_train, df_test, target_train, target_test, figcolnumber):
    # Create a figure with multiple subplots
    fig = plt.figure(
        figsize=(4 * figcolnumber, 4 * ceil(len(modeldict) / figcolnumber))
    )
    fig.set_facecolor("#F3F3F3")  # Set background color of the figure

    for name, figpos in itertools.zip_longest(modeldict, range(len(modeldict))):
        # Define subplot positions
        plt.subplot(ceil(len(modeldict) / figcolnumber), figcolnumber, figpos + 1)

        # Train the model
        model = modeldict[name]
        model.fit(df_train, target_train)

        # Predict probabilities for the test set
        y_prob = model.predict_proba(df_test)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(target_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {name}")
        plt.legend(loc="lower right")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Call the function with your data
rocplot(
    modeldict=models,
    df_train=X_train,
    df_test=X_test,
    target_train=y_train,
    target_test=y_test,
    figcolnumber=3,
)


#
# Best Model:
#  - The Bagging Classifier has the highest AUC score of 0.85, making it the best-performing model in this comparison.
#
# - However, XGBoost and LightGBM are very close with AUC = 0.84, so they also demonstrate excellent performance and can be considered strong alternatives.

# In[84]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# List to store model performance
model_auc_results = []

# Plot ROC Curves for each model
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict the probabilities for the test data (important for ROC AUC)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    if y_prob is not None:
        # Compute the ROC AUC score
        auc_score = roc_auc_score(y_test, y_prob)

        # Get the false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

        # Append the model results
        model_auc_results.append({"Model": name, "AUC": auc_score})

# Add a diagonal line (representing random chance)
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")

# Set plot labels and title
plt.title("ROC Curve for Various Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Create a DataFrame for AUC results
auc_results_df = pd.DataFrame(model_auc_results)
auc_results_df = auc_results_df.round(4)

# Display AUC results
print(auc_results_df)


# From the ROC Curve and AUC (Area Under the Curve) values shown in the plot:
#
# - Bagging Classifier, XGBoost, and LightGBM achieve the highest AUC = 0.84, making them the best-performing models based on ROC-AUC.
# - Gradient Boosting follows closely with an AUC = 0.82.
# Models like Decision Tree and Random Forest perform slightly lower, with AUC = 0.81.
#
#   Final Recommendation:
# - Bagging Classifier, XGBoost, and LightGBM are the best models based on their AUC scores.
#

# ## Model Selection

# Final Conclusion:
#
# Best Model: XGBoost
#
# - Balanced Accuracy, Recall, Precision, and f1-score.
# - Very competitive ROC-AUC score (0.8432).
# - Good generalization and low misclassification rates.
# Close Competitors:
#
# - Bagging Classifier (Highest AUC but lower Recall).
# - LightGBM (Balanced performance but slightly lower AUC).

# ##Deployment

# In[99]:


filename_bagging = "bagging_model.sav"
pickle.dump(bagging_model, open(filename_bagging, "wb"))

filename_lgbm = "lgbm_model.sav"
pickle.dump(lgbm_model, open(filename_lgbm, "wb"))

filename_gbm = "gbm_model.sav"
pickle.dump(gbm_model, open(filename_gbm, "wb"))

filename_xgb = "xgb_model.sav"
pickle.dump(xgb_model, open(filename_xgb, "wb"))

print("All models saved successfully!")


#

# ![465177391_3016241848527895_7850406552759996204_n.png](data:image/png;base64,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)

# ![462577725_1210269707157554_890910394179657262_n.png](data:image/png;base64,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)
