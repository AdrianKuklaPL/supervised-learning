# Necessary imports
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split, cross_val_predict, GridSearchCV, validation_curve, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, make_scorer, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.calibration import cross_val_predict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time


# Set the seed for reproducibility
seed = 42
np.random.seed(seed)         # For NumPy operations
random.seed(seed)            # For Python's built-in random operations

# fetch dataset - drug consumption (quantified)
drug_consumption_quantified = fetch_ucirepo(id=373) 
  
# data (as pandas dataframes)
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets 
  
# metadata 
# print(drug_consumption_quantified.metadata) 
  
# variable information 
# print(drug_consumption_quantified.variables) 

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
columns_X = ["age","gender","education","country","ethnicity","nscore","escore","oscore","ascore",
             "cscore","impulsive","ss"]
X_scaled = pd.DataFrame(X_scaled, columns=columns_X)

# Create a binary classification problem
class_mapping = {
    'CL0': 'Non-user',
    'CL1': 'Non-user',
    'CL2': 'User',
    'CL3': 'User',
    'CL4': 'User',
    'CL5': 'User',
    'CL6': 'User'
}

# Apply the mapping to each target column
y.loc[:, 'cannabis'] = y['cannabis'].map(class_mapping)
Y_Cannabis = y['cannabis']
Y = Y_Cannabis.values.ravel()
Y = np.where(Y == 'Non-user', 0, 1)

pd.set_option('display.max_columns', None)  # Show all columns
print(Y)
pd.reset_option('display.max_columns')


############# Validation curve KNN for different k values ############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define SMOTE and KNN within a pipeline
smote = SMOTE(random_state=seed)
knn = KNeighborsClassifier(weights='uniform', metric='manhattan')

# Create a pipeline that applies SMOTE before fitting KNN
pipeline = Pipeline([('smote', smote), ('knn', knn)])

# Define F1 score as the scoring metric (using weighted average)
f1_scorer = make_scorer(recall_score, average='weighted')

# Define range of k values to evaluate
k_values = [19, 21, 23, 25, 27, 29, 31]
# k_values = [1, 3,  5, 10, 15, 20, 25, 40, 50, 60, 70, 80, 90, 100]

# Perform validation curve calculation using the pipeline with SMOTE and KNN
train_scores, val_scores = validation_curve(
    pipeline, X_train, Y_train, param_name='knn__n_neighbors', param_range=k_values, cv=10, scoring=f1_scorer
)

# Calculate mean and std for train and validation scores
train_mean_f1 = np.mean(train_scores, axis=1)
train_std_f1 = np.std(train_scores, axis=1)
val_mean_f1 = np.mean(val_scores, axis=1)
val_std_f1 = np.std(val_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))

# Plot Training F1 Score with its confidence interval
plt.plot(k_values, train_mean_f1, label='Training F1 Score', marker='o', color='blue', linestyle='dashed')
plt.fill_between(k_values, train_mean_f1 - train_std_f1, train_mean_f1 + train_std_f1, color='blue', alpha=0.2)

# Plot Validation F1 Score with its confidence interval
plt.plot(k_values, val_mean_f1, label='Validation F1 Score', marker='o', color='green', linestyle='solid')
plt.fill_between(k_values, val_mean_f1 - val_std_f1, val_mean_f1 + val_std_f1, color='green', alpha=0.2)

# Add labels to the points for each k value
for i, k in enumerate(k_values):
    plt.text(k_values[i], train_mean_f1[i], f'k={k}', ha='right', va='bottom', fontsize=9, color='blue')
    plt.text(k_values[i], val_mean_f1[i], f'k={k}', ha='right', va='top', fontsize=9, color='green')

# Plot details
plt.title('Validation Curve: F1 Score vs. k (Number of Neighbors) with SMOTE')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()

##### Validation curve KNN for different distance metrics#####
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='weighted')

# Different distance metrics to evaluate
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

# Set k value
k_value = 23
best_weights = 'uniform'

# Initialize lists to store results for each metric
train_f1_scores = []
val_f1_scores = []

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Loop through different distance metrics and use validation curve
for metric in metrics:
    # Initialize KNN model with k=13 and the current distance metric
    knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric, weights=best_weights)
    
    # Create a pipeline with SMOTE and KNN
    pipeline = make_pipeline(smote, knn)
    
    # Perform validation curve calculation
    train_scores, val_scores = validation_curve(
        pipeline, X_train, Y_train, param_name='kneighborsclassifier__n_neighbors',
        param_range=[k_value], cv=10, scoring=f1_scorer
    )
    
    # Calculate mean F1 score for training and validation sets
    train_mean_f1 = np.mean(train_scores, axis=1)[0]  # Single k_value, hence [0]
    val_mean_f1 = np.mean(val_scores, axis=1)[0]

    # Store the results
    train_f1_scores.append(train_mean_f1)
    val_f1_scores.append(val_mean_f1)

    # Output progress
    print(f"Metric={metric}: Train F1={train_mean_f1:.4f}, Validation F1={val_mean_f1:.4f}")

# Plot the validation curve for F1 score
plt.figure(figsize=(8, 6))

# Plot F1 score curves
plt.plot(metrics, train_f1_scores, label='Training F1 Score', marker='o', linestyle='solid', color='blue')
plt.plot(metrics, val_f1_scores, label='Validation F1 Score', marker='o', linestyle='solid', color='green')
plt.title('Validation Curve: F1 Score for Different Distance Metrics with SMOTE')
plt.xlabel('Distance Metric')
plt.ylabel('F1 Score')
plt.legend(loc='best')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()


######## Learning curve KNN ########
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize SMOTE object
smote = SMOTE(random_state=seed)

# Initialize lists to store F1 scores
train_sizes = np.arange(0.1, 1.1, 0.1)  # Training sizes from 10% to 100%
train_f1_scores = []
test_f1_scores = []

# Best model parameters (already identified)
best_k = 23
best_metric = 'manhattan'
best_weight = 'uniform'

# Loop through different training sizes (from 10% to 100%)
for train_size in train_sizes:
    # Cast train_size to float for compatibility with train_test_split
    train_size = float(train_size)

    # If train_size is 1.0, skip the test set since there is no data left for testing
    if train_size < 1.0:
        # Split the data with the current training size
        X_train_partial, X_test_partial, Y_train_partial, Y_test_partial = train_test_split(
            X_scaled, Y, train_size=train_size, random_state=seed)
    else:
        # When train_size is 1.0, use the entire dataset for training
        X_train_partial, Y_train_partial = X_scaled, Y
        X_test_partial, Y_test_partial = None, None

    # Initialize KNN model with the best parameters
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weight)

    # Create a pipeline with SMOTE and KNN
    pipeline = make_pipeline(smote, knn)

    # Train the model on the partial training set
    pipeline.fit(X_train_partial, Y_train_partial)

    # Predict on the training set
    Y_train_pred = pipeline.predict(X_train_partial)

    # Calculate F1 score for the training set
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='macro')

    # Append the results for the training set
    train_f1_scores.append(train_f1)
    
    if train_size < 1.0:
        # Predict on the test set only if it's not the full training set
        Y_test_pred = pipeline.predict(X_test_partial)

        # Calculate F1 score for the test set
        test_f1 = f1_score(Y_test_partial, Y_test_pred, average='macro')

        # Append the results for the test set
        test_f1_scores.append(test_f1)
    else:
        # For train_size=1.0, append None for test F1 scores
        test_f1_scores.append(None)

    # Output progress
    print(f"Train size: {int(train_size*100)}%, Train F1: {train_f1:.4f}")
    if train_size < 1.0:
        print(f"Test F1: {test_f1:.4f}")

# Plotting the learning curve for F1 score only
plt.figure(figsize=(8, 6))

# F1 Score curve
plt.plot(train_sizes * 100, train_f1_scores, label='Training F1 Score', marker='o', color='blue')
plt.plot([size * 100 for size in train_sizes if size < 1.0], [f1 for f1 in test_f1_scores if f1 is not None], label='Validation F1 Score', marker='o', color='green')
plt.xlabel('Training Size (%)')
plt.ylabel('F1 Score')
plt.title('Learning Curve - F1 Score with SMOTE')
plt.legend()

plt.tight_layout()
plt.grid(True)
plt.show()

######## KNN without SMOTE ########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize lists to store results
results = []

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # Different distance metrics to evaluate
weights_options = ['uniform', 'distance']  # Weighting options for neighbors

# Define F1 score as the scoring metric (weighted to handle imbalanced classes)
f1_scorer = make_scorer(f1_score, average='weighted')

# Loop through different k values, distance metrics, and weightings
for metric in metrics:  # Loop through each distance metric
    for weight in weights_options:  # Loop through each weighting option
        for k in k_values:  # Loop through each k value
            
            # Initialize KNN model with current k value, distance metric, and weighting
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)

            # Step 2b: Perform cross-validation using 10 folds with F1 score
            kf = KFold(n_splits=10, shuffle=True, random_state=seed)
            
            # Store F1 scores for each fold
            fold_f1_scores = cross_val_score(knn, X_train, Y_train, cv=kf, scoring=f1_scorer)

            # Calculate and store the mean F1 score across 10 folds
            mean_f1_score = np.mean(fold_f1_scores)

            # Train the model on the entire training set and predict on the test set
            knn.fit(X_train, Y_train)
            Y_pred = knn.predict(X_test)
            
            # Step 3: Store confusion matrix, classification report, and evaluation details
            classification_report_str = classification_report(Y_test, Y_pred)
            
            # Append the result tuple (k, metric, weighting, mean F1 score, classification report)
            results.append((k, metric, weight, mean_f1_score, classification_report_str))

            # Output progress
            print(f"k={k}, metric={metric}, weight={weight}: Mean F1 Score={mean_f1_score:.4f}")

# Find the best combination of k, distance metric, and weighting based on F1 score
best_result = max(results, key=lambda x: x[3])  # Sort by mean F1 score (index 3)
best_k, best_metric, best_weight, best_f1_score, best_classification_report = best_result

# Display the best k, metric, weight, and F1 score
print(f"\nBest k: {best_k}, Best Metric: {best_metric}, Best Weight: {best_weight} with Mean F1 Score: {best_f1_score:.4f}")

# Print final classification report
print(f"\nClassification Report for Best k={best_k}, Metric={best_metric}, Weight={best_weight}:\n{best_classification_report}")


########### KNN WITH SMOTE #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the fixed parameters
k = 23
metric = 'manhattan'
weight = 'uniform'

# Define F1 score as the scoring metric (weighted)
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Create a pipeline with SMOTE and KNN
knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
pipeline = Pipeline([('smote', smote), ('knn', knn)])  # Use imblearn's Pipeline to integrate SMOTE

# Perform cross-validation using 10 folds with F1 score
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring=f1_scorer)

# Calculate and store the mean F1 score across 10 folds
mean_f1_score_train = np.mean(cv_scores)

# Train the model on the entire training set
pipeline.fit(X_train, Y_train)

# Make predictions on the full training set
Y_train_pred = pipeline.predict(X_train)
train_f1_score = f1_score(Y_train, Y_train_pred, average='weighted')

# Make predictions on the test set
Y_test_pred = pipeline.predict(X_test)
test_f1_score = f1_score(Y_test, Y_test_pred, average='weighted')

# Output the results
print(f"Mean F1 Score (CV on Training Set): {mean_f1_score_train:.4f}")
print(f"F1 Score (Full Training Set): {train_f1_score:.4f}")
print(f"F1 Score (Test Set): {test_f1_score:.4f}")

# Generate and print classification report for test set
classification_report_str = classification_report(Y_test, Y_test_pred)
print("\nClassification Report on Test Set:\n", classification_report_str)



############ Validation Curve SVM with SMOTE ############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define kernel methods
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Fix C value
C_value = 1

# Define F1 score as the scoring metric (using weighted average)
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Initialize lists to store the mean and standard deviation of F1 scores
train_mean_f1_scores = []
train_std_f1_scores = []
val_mean_f1_scores = []
val_std_f1_scores = []

# Perform validation curve calculation for each kernel
for kernel in kernels:
    # Initialize the SVM model with the current kernel and fixed C value
    svm = SVC(kernel=kernel, C=C_value, random_state=seed)
    
    # Create a pipeline with SMOTE and SVM
    pipeline = Pipeline([('smote', smote), ('svm', svm)])
    
    # Use validation_curve from sklearn with the pipeline
    train_scores, val_scores = validation_curve(
        pipeline, X_train, Y_train, param_name='svm__C', param_range=[C_value], cv=10, scoring=f1_scorer
    )

    # Calculate mean and std for training and validation F1 scores
    train_mean_f1 = np.mean(train_scores, axis=1)[0]
    train_std_f1 = np.std(train_scores, axis=1)[0]
    val_mean_f1 = np.mean(val_scores, axis=1)[0]
    val_std_f1 = np.std(val_scores, axis=1)[0]

    # Append the results to the lists
    train_mean_f1_scores.append(train_mean_f1)
    train_std_f1_scores.append(train_std_f1)
    val_mean_f1_scores.append(val_mean_f1)
    val_std_f1_scores.append(val_std_f1)

    # Output progress
    print(f"Kernel: {kernel}, Train F1: {train_mean_f1:.4f}, Validation F1: {val_mean_f1:.4f}")

# Plot the validation curve for F1 score

plt.figure(figsize=(8, 6))

# Plot F1 score curves with error bars
plt.plot(kernels, train_mean_f1_scores, label='Training F1 Score', marker='o', linestyle='dashed', color='blue')
plt.fill_between(kernels, 
                 np.array(train_mean_f1_scores) - np.array(train_std_f1_scores), 
                 np.array(train_mean_f1_scores) + np.array(train_std_f1_scores), 
                 color='blue', alpha=0.2)

plt.plot(kernels, val_mean_f1_scores, label='Validation F1 Score', marker='o', linestyle='solid', color='green')
plt.fill_between(kernels, 
                 np.array(val_mean_f1_scores) - np.array(val_std_f1_scores), 
                 np.array(val_mean_f1_scores) + np.array(val_std_f1_scores), 
                 color='green', alpha=0.2)

plt.title('Validation Curve: F1 Score for Different Kernels (C=1) with SMOTE')
plt.xlabel('Kernel')
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

########### Validation Curve: Accuracy for different values of C #########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

############ SVM WITH SMOTE #############
C_values = [0.001, 0.01, 0.1, 1, 10]
kernel = 'rbf'

# Define F1 score as the scoring metric (weighted)
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Initialize the SVM model with a fixed linear kernel and create a pipeline
svm = SVC(kernel=kernel, random_state=seed)
pipeline = Pipeline([('smote', smote), ('svm', svm)])

# Perform validation curve calculation using weighted F1 score
train_scores, val_scores = validation_curve(
    pipeline, X_train, Y_train, param_name='svm__C', param_range=C_values, cv=10, scoring=f1_scorer
)

# Calculate the mean and standard deviation for the F1 scores
train_mean_f1 = np.mean(train_scores, axis=1)
train_std_f1 = np.std(train_scores, axis=1)
val_mean_f1 = np.mean(val_scores, axis=1)
val_std_f1 = np.std(val_scores, axis=1)

# Plot the validation curve for F1 score
plt.figure(figsize=(8, 6))

# Plot F1 score curves with error bars
plt.plot(C_values, train_mean_f1, label='Training F1 Score', marker='o', linestyle='dashed', color='blue')
plt.fill_between(C_values, train_mean_f1 - train_std_f1, train_mean_f1 + train_std_f1, color='blue', alpha=0.2)

plt.plot(C_values, val_mean_f1, label='Validation F1 Score', marker='o', linestyle='solid', color='green')
plt.fill_between(C_values, val_mean_f1 - val_std_f1, val_mean_f1 + val_std_f1, color='green', alpha=0.2)

plt.title('Validation Curve: F1 Score for Different C values (Linear Kernel) with SMOTE')
plt.xlabel('C Value')
plt.xscale('log')  # Use logarithmic scale for C
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

############ Learning curve SVM with SMOTE ############
# Split the data into 80% training and 20% testing (initial split)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the best model parameters
best_kernel = 'linear'
best_C = 1

# Define different training sizes (from 10% to 100%)
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize lists to store F1 scores
train_f1_scores = []
test_f1_scores = []

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Loop through different training sizes
for train_size in train_sizes:
    
    # Split the current X_train and Y_train into partial training and testing sets
    if train_size == 1.0:
        # Use the full training set if train_size is 100%
        X_train_partial, Y_train_partial = X_train, Y_train
        X_test_partial, Y_test_partial = None, None  # No test set
    else:
        # Split the data into partial training and testing sets
        X_train_partial, X_test_partial, Y_train_partial, Y_test_partial = train_test_split(
            X_train, Y_train, train_size=train_size, random_state=seed)

    # Create a pipeline with SMOTE and SVM
    pipeline = Pipeline([('smote', smote), ('svm', SVC(kernel=best_kernel, C=best_C, random_state=seed))])
    
    # Train the model on the partial training set
    pipeline.fit(X_train_partial, Y_train_partial)

    # Make predictions on the partial training set
    Y_train_pred = pipeline.predict(X_train_partial)

    # Calculate F1 score for the partial training set
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
    train_f1_scores.append(train_f1)

    # If test data exists, evaluate on the partial test set
    if X_test_partial is not None:
        Y_test_pred = pipeline.predict(X_test_partial)
        test_f1 = f1_score(Y_test_partial, Y_test_pred, average='weighted')
    else:
        test_f1 = None

    # Store test F1 score (or None if no test set)
    test_f1_scores.append(test_f1)

    # Output progress
    print(f"Training Size: {train_size*100:.0f}%, Train F1: {train_f1:.4f}, Test F1: {test_f1 if test_f1 is not None else 'N/A'}")

# Plot the learning curve for F1 score

plt.figure(figsize=(8, 6))

# Plot F1 score curves
plt.plot(train_sizes, train_f1_scores, label='Training F1 Score', marker='o', color='blue')
plt.plot(train_sizes[:-1], [f1 for f1 in test_f1_scores if f1 is not None], label='Validation F1 Score', marker='o', color='green')

plt.title('Learning Curve: F1 Score as a Function of Training Size with SMOTE')
plt.xlabel('Training Size')
plt.ylabel('F1 Score')
plt.legend(loc='best')

plt.tight_layout()
plt.grid(True)
plt.show()

############ SVM WITHOUT SMOTE - Grid Search #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize vectors of different kernel methods and different values for C
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.001,0.01, 0.1, 1, 10]

# Create KFold cross-validation object
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Define F1 score as the scoring metric (using weighted average)
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize lists to store model results
model_results = []  # To store (kernel, C, mean_f1_score)
classification_reports = {}  # To store classification reports for each model

# Perform cross-validation for each combination of kernel and C
for kernel in kernels:
    for C in C_values:
        # Initialize the SVM model with the current kernel and C value
        svm = SVC(kernel=kernel, C=C, random_state=seed)

        # Perform 10-fold cross-validation using the F1 score as the scoring metric
        cv_scores = cross_val_score(svm, X_train, Y_train, cv=kf, scoring=f1_scorer)

        # Store the mean F1 score across 10 folds
        mean_f1_score = np.mean(cv_scores)
        model_results.append((kernel, C, mean_f1_score))  # Store kernel, C, and F1 score
        
        # Train the model on the entire training set
        svm.fit(X_train, Y_train)

        # Predict on the test set
        Y_pred = svm.predict(X_test)
        
        # Store the classification report for each kernel and C combination
        report = classification_report(Y_test, Y_pred)
        classification_reports[(kernel, C)] = report  # Store the report in the dictionary
        print(f"\nClassification Report for Kernel: {kernel}, C: {C}:\n")
        print(report)

        # Output progress
        print(f"Kernel: {kernel}, C: {C}, Mean F1 Score: {mean_f1_score:.4f}")

# Output the stored model results (kernel, C, and corresponding F1 score)
print("\nAll Model Results (Kernel, C, Mean F1 Score):")
for result in model_results:
    print(f"Kernel: {result[0]}, C: {result[1]}, Mean F1 Score: {result[2]:.4f}")

# Find the best model based on the highest mean F1 score
best_model = max(model_results, key=lambda x: x[2])
best_kernel, best_C, best_f1_score = best_model

# Output the best model and its classification report
print(f"\nBest Model -> Kernel: {best_kernel}, C: {best_C}, F1 Score: {best_f1_score:.4f}")
print("\nClassification Report for the Best Model:\n")
print(classification_reports[(best_kernel, best_C)])

############ SVM WITH SMOTE #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

kernel = 'linear'
C_value = 1

# Create KFold cross-validation object
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Define F1 score as the scoring metric (using weighted average)
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize the SVM model with RBF kernel and C = 0.1
svm = SVC(kernel=kernel, C=C_value, random_state=seed)

# Create a pipeline to apply SMOTE followed by the SVM classifier
smote = SMOTE(random_state=seed)
pipeline = Pipeline([('smote', smote), ('svm', svm)])

# Perform 10-fold cross-validation using the F1 score on training data
cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring=f1_scorer)

# Store the mean F1 score across 10 folds for training data evaluation
mean_f1_score_train = np.mean(cv_scores)

# Train the pipeline on the full training set
pipeline.fit(X_train, Y_train)

# Calculate F1 score on the full training set
Y_train_pred = pipeline.predict(X_train)
f1_score_train = f1_score(Y_train, Y_train_pred, average='weighted')

# Predict on the test set
Y_test_pred = pipeline.predict(X_test)

# Calculate F1 score on the test set
f1_score_test = f1_score(Y_test, Y_test_pred, average='weighted')

# Generate classification report on the test set
report = classification_report(Y_test, Y_test_pred)

# Output results
print(f"Mean F1 Score (CV on Training Set): {mean_f1_score_train:.4f}")
print(f"F1 Score (Full Training Set): {f1_score_train:.4f}")
print(f"F1 Score (Test Set): {f1_score_test:.4f}")
print("\nClassification Report (Test Set):\n")
print(report)

######## Neural Networks ########
######## Neural Network with CV #######
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define KFold cross-validation with 5 splits (can adjust based on your needs)
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Define the neural network model (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=seed, max_iter=500)

# Perform cross-validation and get predictions for each fold on the training data
Y_train_pred = cross_val_predict(model, X_train, Y_train, cv=kf)

# Train the model on the full training set and evaluate on the test set
model.fit(X_train, Y_train)
Y_test_pred = model.predict(X_test)

# Calculate accuracy and F1 score on cross-validated training data
train_accuracy = accuracy_score(Y_train, Y_train_pred)
train_f1 = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate accuracy and F1 score on the test data
test_accuracy = accuracy_score(Y_test, Y_test_pred)
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

# Print performance metrics for training and test sets
print(f"Cross-validated Training Accuracy: {train_accuracy:.4f}")
print(f"Cross-validated Training F1 Score: {train_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

### Note that the performance when using cross validation reduced for the training data set.
### This is likely due to cross validation improving the generalization of the model.

#### Neural network with CV and SMOTE ####
# Initialize SMOTE and MLPClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
smote = SMOTE(random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=seed, max_iter=500)

# Combine SMOTE and the classifier in a pipeline
pipeline = make_pipeline(smote, model)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform cross-validation with SMOTE applied to the training set in each fold
Y_train_pred = cross_val_predict(pipeline, X_train, Y_train, cv=kf)

# Train the model on the full training set and evaluate on the test set
pipeline.fit(X_train, Y_train)
Y_test_pred = pipeline.predict(X_test)

# Calculate accuracy and F1 score on the cross-validated training data
train_accuracy = accuracy_score(Y_train, Y_train_pred)
train_f1 = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate accuracy and F1 score on the test data
test_accuracy = accuracy_score(Y_test, Y_test_pred)
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

# Print performance metrics for training and test sets
print(f"Cross-validated Training Accuracy: {train_accuracy:.4f}")
print(f"Cross-validated Training F1 Score: {train_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

### Reduced the training accuracy and F1 score but improved the test accuracy and F1 score.


##### Neural Network Validation Curve for Activation Function #####
# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Different activation functions to test
activation_functions = ['relu', 'logistic', 'tanh']

# Lists to store mean F1 scores for training and validation sets for each activation function
train_f1_means = []
val_f1_means = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Iterate over different activation functions
for activation in activation_functions:
    # Define the MLPClassifier with varying activation functions
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation=activation, solver='adam', random_state=seed
                          ,alpha=0.001, max_iter=500)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Initialize lists to store F1 scores for each fold
    train_f1_fold_scores = []
    val_f1_fold_scores = []

    # Perform K-Fold cross-validation
    for train_index, val_index in kf.split(X_train):
        # Extract partial training and validation sets using iloc for X_train and numpy indexing for Y_train
        X_train_partial, X_val_partial = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train_partial, Y_val_partial = Y_train[train_index], Y_train[val_index]
        
        # Fit the pipeline to the partial training set
        pipeline.fit(X_train_partial, Y_train_partial)
        
        # Predict on the partial training and validation sets
        Y_train_pred = pipeline.predict(X_train_partial)
        Y_val_pred = pipeline.predict(X_val_partial)
        
        # Calculate F1 scores (weighted) for both partial training and validation sets
        train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='weighted')
        
        # Store F1 scores for this fold
        train_f1_fold_scores.append(train_f1)
        val_f1_fold_scores.append(val_f1)

    # Calculate mean F1 scores for this activation function across all folds
    mean_train_f1 = np.mean(train_f1_fold_scores)
    mean_val_f1 = np.mean(val_f1_fold_scores)
    
    # Store the mean F1 scores
    train_f1_means.append(mean_train_f1)
    val_f1_means.append(mean_val_f1)

    # Output progress
    print(f"Activation: {activation}, Mean Train F1: {mean_train_f1:.4f}, Mean Validation F1: {mean_val_f1:.4f}")

# Plotting the validation curve for F1 score for each activation function
plt.figure(figsize=(10, 6))
plt.plot(activation_functions, train_f1_means, label="Mean Training F1 Score", marker='o', color='blue')
plt.plot(activation_functions, val_f1_means, label="Mean Validation F1 Score", marker='o', color='green')
plt.title("Validation Curve - F1 Score (Weighted) vs. Activation Functions")
plt.xlabel("Activation Function")
plt.ylabel("Mean F1 Score (Weighted)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Validation Curve for Learning Rate in Neural Network with SMOTE ##########
# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Different learning rates to test
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# Lists to store mean F1 scores for training and validation sets for each learning rate
train_f1_means = []
val_f1_means = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Iterate over different learning rates
for lr in learning_rates:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', learning_rate_init=lr, 
                          random_state=seed, alpha=0.001, max_iter=500)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Initialize lists to store F1 scores for each fold
    train_f1_fold_scores = []
    val_f1_fold_scores = []

    # Perform K-Fold cross-validation
    for train_index, val_index in kf.split(X_train):
        # Extract partial training and validation sets using iloc for X_train and numpy indexing for Y_train
        X_train_partial, X_val_partial = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train_partial, Y_val_partial = Y_train[train_index], Y_train[val_index]
        
        # Fit the pipeline to the partial training set
        pipeline.fit(X_train_partial, Y_train_partial)
        
        # Predict on the partial training and validation sets
        Y_train_pred = pipeline.predict(X_train_partial)
        Y_val_pred = pipeline.predict(X_val_partial)
        
        # Calculate F1 scores (weighted) for both partial training and validation sets
        train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='weighted')
        
        # Store F1 scores for this fold
        train_f1_fold_scores.append(train_f1)
        val_f1_fold_scores.append(val_f1)

    # Calculate mean F1 scores for this learning rate across all folds
    mean_train_f1 = np.mean(train_f1_fold_scores)
    mean_val_f1 = np.mean(val_f1_fold_scores)
    
    # Store the mean F1 scores
    train_f1_means.append(mean_train_f1)
    val_f1_means.append(mean_val_f1)

    # Output progress
    print(f"Learning Rate: {lr}, Mean Train F1: {mean_train_f1:.4f}, Mean Validation F1: {mean_val_f1:.4f}")

# Plotting the validation curve for F1 score for each learning rate
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, train_f1_means, label="Mean Training F1 Score", marker='o', color='blue')
plt.plot(learning_rates, val_f1_means, label="Mean Validation F1 Score", marker='o', color='green')
plt.title("Validation Curve - F1 Score (Weighted) vs. Learning Rates")
plt.xlabel("Learning Rate")
plt.ylabel("Mean F1 Score (Weighted)")
plt.xscale('log')  # Logarithmic scale for learning rates
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Validation Curve for Batch Size in Neural Network with SMOTE ##########
# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Different batch sizes to test
batch_sizes = [32, 64, 128, 256, 512]

# Lists to store mean F1 scores for training and validation sets for each batch size
train_f1_means = []
val_f1_means = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Iterate over different batch sizes
for batch_size in batch_sizes:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', learning_rate_init=0.0001, 
                          random_state=seed, batch_size=batch_size, alpha=0.001, max_iter=500)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Initialize lists to store F1 scores for each fold
    train_f1_fold_scores = []
    val_f1_fold_scores = []

    # Perform K-Fold cross-validation
    for train_index, val_index in kf.split(X_train):
        # Extract partial training and validation sets using iloc for X_train and numpy indexing for Y_train
        X_train_partial, X_val_partial = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train_partial, Y_val_partial = Y_train[train_index], Y_train[val_index]
        
        # Fit the pipeline to the partial training set
        pipeline.fit(X_train_partial, Y_train_partial)
        
        # Predict on the partial training and validation sets
        Y_train_pred = pipeline.predict(X_train_partial)
        Y_val_pred = pipeline.predict(X_val_partial)
        
        # Calculate F1 scores (weighted) for both partial training and validation sets
        train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='weighted')
        
        # Store F1 scores for this fold
        train_f1_fold_scores.append(train_f1)
        val_f1_fold_scores.append(val_f1)

    # Calculate mean F1 scores for this batch size across all folds
    mean_train_f1 = np.mean(train_f1_fold_scores)
    mean_val_f1 = np.mean(val_f1_fold_scores)
    
    # Store the mean F1 scores
    train_f1_means.append(mean_train_f1)
    val_f1_means.append(mean_val_f1)

    # Output progress
    print(f"Batch Size: {batch_size}, Mean Train F1: {mean_train_f1:.4f}, Mean Validation F1: {mean_val_f1:.4f}")

# Plotting the validation curve for F1 score for each batch size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, train_f1_means, label="Mean Training F1 Score", marker='o', color='blue')
plt.plot(batch_sizes, val_f1_means, label="Mean Validation F1 Score", marker='o', color='green')
plt.title("Validation Curve - F1 Score (Weighted) vs. Batch Sizes")
plt.xlabel("Batch Size")
plt.ylabel("Mean F1 Score (Weighted)")
plt.xscale('log')  # Logarithmic scale for batch sizes
plt.xticks(batch_sizes, labels=batch_sizes)  # Ensure the batch sizes are displayed as ticks
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()


######## Validation Curve for Hidden Layers in Neural Network with SMOTE ##########
# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Different architectures to test: 1-layer and 2-layer with varying neurons
layer_configurations = [
    (16,),     # 1 layer with 16 neurons
    (32,),     # 1 layer with 32 neurons
    (64,),     # 1 layer with 64 neurons
    (128,),    # 1 layer with 128 neurons
    (32, 16),  # 2 layers, both with 16 neurons
    (64, 32),  # 2 layers, both with 32 neurons
    (128, 64),  # 2 layers, both with 64 neurons
    (256, 128),  # 2 layers, both with 128 neurons
    (64, 32, 16),
    (128, 64, 32),
]

# Lists to store mean F1 scores for training and validation sets for each configuration
train_f1_means = []
val_f1_means = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Iterate over different layer configurations
for layer_config in layer_configurations:
    # Define the MLPClassifier with varying layer configurations
    model = MLPClassifier(hidden_layer_sizes=layer_config, activation='tanh', solver='adam', 
                          random_state=seed, alpha=0.001,learning_rate_init=0.0001,batch_size=128, max_iter=500)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Initialize lists to store F1 scores for each fold
    train_f1_fold_scores = []
    val_f1_fold_scores = []

    # Perform K-Fold cross-validation
    for train_index, val_index in kf.split(X_train):
        # Extract partial training and validation sets using iloc for X_train and numpy indexing for Y_train
        X_train_partial, X_val_partial = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train_partial, Y_val_partial = Y_train[train_index], Y_train[val_index]
        
        # Fit the pipeline to the partial training set
        pipeline.fit(X_train_partial, Y_train_partial)
        
        # Predict on the partial training and validation sets
        Y_train_pred = pipeline.predict(X_train_partial)
        Y_val_pred = pipeline.predict(X_val_partial)
        
        # Calculate F1 scores (weighted) for both partial training and validation sets
        train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='weighted')
        
        # Store F1 scores for this fold
        train_f1_fold_scores.append(train_f1)
        val_f1_fold_scores.append(val_f1)

    # Calculate mean F1 scores for this configuration across all folds
    mean_train_f1 = np.mean(train_f1_fold_scores)
    mean_val_f1 = np.mean(val_f1_fold_scores)
    
    # Store the mean F1 scores
    train_f1_means.append(mean_train_f1)
    val_f1_means.append(mean_val_f1)

    # Output progress
    print(f"Layer Configuration: {layer_config}, Mean Train F1: {mean_train_f1:.4f}, Mean Validation F1: {mean_val_f1:.4f}")

# Plotting the validation curve for F1 score for each layer configuration
plt.figure(figsize=(10, 6))
layer_labels = [str(layer) for layer in layer_configurations]
plt.plot(layer_labels, train_f1_means, label="Mean Training F1 Score", marker='o', color='blue')
plt.plot(layer_labels, val_f1_means, label="Mean Validation F1 Score", marker='o', color='green')
plt.title("Validation Curve - F1 Score (Weighted) vs. Layer Configurations")
plt.xlabel("Layer Configurations (Neurons)")
plt.ylabel("Mean F1 Score (Weighted)")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Learning curve for Neural Network with SMOTE #########
# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
## Train-test split size ranges from 10% to 100% for the learning curve
train_sizes = np.arange(0.1, 1, 0.1)  # Training sizes from 10% to 90%

# Initialize lists to store F1 scores
train_f1_scores = []
test_f1_scores = []

# Split the initial dataset into training and testing (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Loop through different training sizes for learning curve calculation
for train_size in train_sizes:
    # Split X_train and Y_train into partial sets based on the current train_size
    X_train_partial, X_test_partial, Y_train_partial, Y_test_partial = train_test_split(
        X_train, Y_train, train_size=train_size, random_state=seed
    )
    
    # Initialize SMOTE and MLPClassifier (Neural Network)
    smote = SMOTE(random_state=seed)
    model = MLPClassifier(hidden_layer_sizes=(32,16), activation='tanh', solver='adam', 
                          random_state=seed, alpha=0.001,learning_rate_init=0.0001,batch_size=128, max_iter=500)
    # Create a pipeline with SMOTE and the MLPClassifier
    pipeline = make_pipeline(smote, model)
    
    # Train the model on the partial training set
    pipeline.fit(X_train_partial, Y_train_partial)
    
    # Predict on both partial training set and test partial set
    Y_train_pred = pipeline.predict(X_train_partial)
    Y_test_pred = pipeline.predict(X_test_partial)
    
    # Calculate F1 score (weighted) on both training and testing sets
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
    test_f1 = f1_score(Y_test_partial, Y_test_pred, average='weighted')
    
    # Store the F1 scores for plotting
    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)

    # Output progress
    print(f"Train Size: {train_size*100:.0f}%, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

# Plot Learning Curves: F1 Score
plt.figure(figsize=(12, 6))
plt.plot(train_sizes * 100, train_f1_scores, label="Training F1 Score", marker='o', color='blue')
plt.plot(train_sizes * 100, test_f1_scores, label="Testing F1 Score", marker='o', color='green')
plt.title("Learning Curve - F1 Score vs. Training Size")
plt.xlabel("Training Size (%)")
plt.ylabel("F1 Score (Weighted)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Learning Curve for Neural Network with SMOTE (Number of Epochs) ##########

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize SMOTE and set up neural network with warm_start for incremental learning
smote = SMOTE(random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(32,16), activation='tanh', solver='adam',random_state=seed,
                       alpha=0.001,learning_rate_init=0.0001,batch_size=128, max_iter=500, warm_start=True, early_stopping=True)
    
# Combine SMOTE and the classifier in a pipeline
pipeline = make_pipeline(smote, model)

# Lists to store F1 scores over epochs
train_f1_scores = []
test_f1_scores = []

# Define the number of epochs (iterations)
n_epochs = 500

# Train the model for n_epochs and record performance at each epoch
for epoch in range(n_epochs):
    # Split X_train and Y_train into partial sets based on the current epoch
    X_train_partial, X_test_partial, Y_train_partial, Y_test_partial = train_test_split(
        X_train, Y_train, train_size=0.8, random_state=seed
    )
    
    # Train for one epoch
    pipeline.fit(X_train_partial, Y_train_partial)
    
    # Predict on both partial training set and test partial set
    Y_train_pred = pipeline.predict(X_train_partial)
    Y_test_pred = pipeline.predict(X_test_partial)
    
    # Calculate F1 score (weighted) on both partial training and testing sets
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='weighted')
    test_f1 = f1_score(Y_test_partial, Y_test_pred, average='weighted')
    
    # Store the metrics for plotting
    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)

    # Output progress
    print(f"Epoch {epoch+1}/{n_epochs}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

# Plot Learning Curve: F1 Score
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_epochs + 1), train_f1_scores, label="Training F1 Score", marker='o', linestyle='-', color='blue')
plt.plot(range(1, n_epochs + 1), test_f1_scores, label="Validation F1 Score", marker='o', linestyle='-', color='green')
plt.title("Learning Curve - F1 Score vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1 Score (Weighted)")
plt.legend(loc="best")
plt.grid(True)
plt.show()


####### Grid Search #######
# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Create the pipeline with SMOTE and the MLPClassifier
pipeline = make_pipeline(smote, MLPClassifier(random_state=seed, max_iter=500))

# Define the parameter grid for GridSearch
param_grid = {
    'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'mlpclassifier__learning_rate_init': [0.001, 0.01, 0.1],    # Learning rates
    'mlpclassifier__solver': ['adam', 'sgd'],                   # Solvers
    'mlpclassifier__hidden_layer_sizes': [(64, 32), (128, 64)],  # Hidden layers
    'mlpclassifier__alpha': [0.0001, 0.001, 0.01]               # L2 penalty (regularization term)
}

# Define the F1 score (weighted) as the scoring metric
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=f1_scorer, n_jobs=-1)

# Perform the grid search on the training set
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Output the best parameters and corresponding F1 score
print("Best Parameters: ", best_params)
print("Best F1 Score on Training Data: ", grid_search.best_score_)

# Evaluate the best model on the test set
Y_test_pred = best_model.predict(X_test)
test_f1 = f1_score(Y_test, Y_test_pred, average='weighted')
print("F1 Score on Test Data: ", test_f1)


# Best Parameters:  {'mlpclassifier__activation': 'logistic', 'mlpclassifier__alpha': 0.01,
# 'mlpclassifier__hidden_layer_sizes': (64, 32), 'mlpclassifier__learning_rate_init': 0.1,
#  'mlpclassifier__solver': 'sgd'}
# Best F1 Score on Training Data:  0.8028112404929848
# F1 Score on Test Data:  0.7985786522704286

######## Selected Neural Network Model #######
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
smote = SMOTE(random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam', 
                      random_state=seed,batch_size=128, learning_rate_init=0.0001,
                      alpha=0.001, max_iter=500)

# Combine SMOTE and the classifier in a pipeline
pipeline = make_pipeline(smote, model)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform cross-validation with SMOTE applied to the training set in each fold
Y_train_pred = cross_val_predict(pipeline, X_train, Y_train, cv=kf)

# Train the model on the full training set and evaluate on the test set
pipeline.fit(X_train, Y_train)
Y_test_pred = pipeline.predict(X_test)

# Calculate weighted F1 score on the cross-validated training data
train_f1_weighted = f1_score(Y_train, Y_train_pred, average='weighted')

# Calculate weighted F1 score on the test data
test_f1_weighted = f1_score(Y_test, Y_test_pred, average='weighted')

# Print performance metrics for training and test sets
print(f"Cross-validated Training F1 Score (Weighted): {train_f1_weighted:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")


######## Measuring Training Time (Clock time and iterations) for SVM ########
# Measure wall clock time
# Best SVM model parameters
# Step 1: Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
best_kernel = 'linear'
best_C = 1

# Measure wall clock time for training
start_time = time.time()
svc = SVC(kernel=best_kernel, C=best_C)
svc.fit(X_train, Y_train)
end_time = time.time()
time_svm_training = round(end_time - start_time, 4)


# Measure wall clock time for prediction
start_time = time.time()
Y_pred_svm = svc.predict(X_test)
end_time = time.time()
time_svm_prediction = round(end_time - start_time, 4)


# To get iteration count, use LinearSVC:
from sklearn.svm import LinearSVC
svc_linear = LinearSVC(max_iter=1000)
svc_linear.fit(X_train, Y_train)



######## Measuring Training Time (Clock time and iterations) for KNN ########
# Best KNN model parameters
best_k = 23
best_metric = 'manhattan'
best_weight = 'uniform'

# Measure wall clock time for training
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weight)
knn.fit(X_train, Y_train)
end_time = time.time()
time_knn_training = round(end_time - start_time, 4)


# Measure wall clock time for prediction
start_time = time.time()
Y_pred_knn = knn.predict(X_test)
end_time = time.time()
time_knn_prediction = round(end_time - start_time, 4)


######## Measuring Training Time (Clock time and iterations) for Neural Network ########
# Best Neural Network parameters
hidden_layers = (32,16)
activation = 'tanh'
solver = 'adam'
max_iter = 500
max_batch = 128
max_learning_rate = 0.0001
max_alpha = 0.001

# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Measure wall clock time for training the neural network
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver,
                    batch_size=max_batch, learning_rate_init=max_learning_rate, alpha=max_alpha,
                     random_state=seed, max_iter=max_iter)
mlp.fit(X_train, Y_train)
end_time = time.time()
time_mlp_training = round(end_time - start_time, 4)

# Measure wall clock time for prediction using the trained model
start_time = time.time()
Y_pred_mlp = mlp.predict(X_test)
end_time = time.time()
time_mlp_prediction = round(end_time - start_time, 4)

# Number of iterations the MLP took to converge
mlp_iterations = mlp.n_iter_

# Print the results
print(f"SVM Number of iterations: {svc_linear.n_iter_}")
print(f"Neural Network (MLPClassifier) Iterations: {mlp_iterations}")

print(f"SVM Training Time (Wall Clock): {time_svm_training} seconds")
print(f"KNN Training Time (Wall Clock): {time_knn_training} seconds")
print(f"Neural Network (MLPClassifier) Training Time: {time_mlp_training} seconds")

print(f"SVM Prediction Time (Wall Clock): {time_svm_prediction} seconds")
print(f"KNN Prediction Time (Wall Clock): {time_knn_prediction} seconds")
print(f"Neural Network (MLPClassifier) Prediction Time: {time_mlp_prediction} seconds")

# KNN Results
# F1 Score on Full Training Set: 0.7999
# F1 Score on Test Set: 0.7735

#SVM Results
# F1 Score (Full Training Set): 0.7933
# F1 Score (Test Set): 0.7937

# Neural Network Results
# Full Training F1 Score: 0.7958
# Test F1 Score: 0.8057

##### End of Drug Consumption Project #####