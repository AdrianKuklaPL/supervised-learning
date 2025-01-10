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

# Fetch dataset - Iranian churn
iranian_churn = fetch_ucirepo(id=563) 

# Data (as pandas dataframes)
X = iranian_churn.data.features
y = iranian_churn.data.targets

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
columns_X = ["Call Failure", "Complains", "Subscription Length", "Charge Amount", "Seconds of Use",
            "Frequency of Use", "Frequency of SMS", "Distinct Called Numbers", "Age Group", "Tariff Plan",
            "Status", "Age", "Customer Value"]
X_scaled = pd.DataFrame(X_scaled, columns=columns_X)
Y = y.values.ravel()

########## Validation Curve for different k values KNN ###########
# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set k values and initialize KNN model
k_values = [1, 3, 5, 7, 9, 11, 15, 19, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65]
knn = KNeighborsClassifier(weights='uniform', metric='manhattan')

# Create a pipeline with SMOTE and KNN
smote_knn_pipeline = Pipeline([
    ('smote', SMOTE(random_state=seed)),
    ('knn', knn)
])

# Use scikit-learn's validation_curve function
# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Compute validation curve using training data with SMOTE included in the pipeline
train_scores, val_scores = validation_curve(
    smote_knn_pipeline, X_train, Y_train, param_name='knn__n_neighbors', param_range=k_values, cv=10, scoring=f1_scorer
)

# Calculate the mean and standard deviation for training and validation scores
train_mean_f1 = np.mean(train_scores, axis=1)
train_std_f1 = np.std(train_scores, axis=1)
val_mean_f1 = np.mean(val_scores, axis=1)
val_std_f1 = np.std(val_scores, axis=1)

# Plot the validation curve for F1 score
plt.figure(figsize=(10, 6))

# Plot training F1 score curve with labels
plt.plot(k_values, train_mean_f1, label='Training F1 Score', marker='o', color='blue', linestyle='solid')
plt.fill_between(k_values, train_mean_f1 - train_std_f1, train_mean_f1 + train_std_f1, color='blue', alpha=0.2)

# Adding k-value labels to training F1 score points
for i, k in enumerate(k_values):
    plt.text(k, train_mean_f1[i], f'{k}', ha='right', va='bottom', fontsize=9)

# Plot validation F1 score curve with labels
plt.plot(k_values, val_mean_f1, label='Validation F1 Score', marker='o', color='green', linestyle='solid')
plt.fill_between(k_values, val_mean_f1 - val_std_f1, val_mean_f1 + val_std_f1, color='green', alpha=0.2)

# Adding k-value labels to validation F1 score points
for i, k in enumerate(k_values):
    plt.text(k, val_mean_f1[i], f'{k}', ha='right', va='bottom', fontsize=9)

# Plot settings
plt.title('Validation Curve: F1 Score vs. k (Number of Neighbors) with SMOTE')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()

############# Validation curve KNN for different distance metrics ############

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

######## KNN Different Distance Metrics ########
# Set distance metrics and initialize KNN model
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
k = 3

# Create a pipeline with SMOTE and KNN
smote_knn_pipeline = Pipeline([
    ('smote', SMOTE(random_state=seed)),
    ('knn', KNeighborsClassifier(n_neighbors=k, weights='uniform'))
])

# Use scikit-learn's validation_curve function
# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Compute validation curve using training data with SMOTE included in the pipeline
train_scores, val_scores = validation_curve(
    smote_knn_pipeline, X_train, Y_train, param_name='knn__metric', param_range=metrics, cv=10, scoring=f1_scorer
)

# Calculate the mean and standard deviation for training and validation scores
train_mean_f1 = np.mean(train_scores, axis=1)
train_std_f1 = np.std(train_scores, axis=1)
val_mean_f1 = np.mean(val_scores, axis=1)
val_std_f1 = np.std(val_scores, axis=1)

# Plot the validation curve for F1 score
plt.figure(figsize=(10, 6))

plt.plot(metrics, train_mean_f1, label='Training F1 Score', marker='o', color='blue', linestyle='dashed')
plt.fill_between(metrics, train_mean_f1 - train_std_f1, train_mean_f1 + train_std_f1, color='blue', alpha=0.2)

plt.plot(metrics, val_mean_f1, label='Validation F1 Score', marker='o', color='green', linestyle='solid')
plt.fill_between(metrics, val_mean_f1 - val_std_f1, val_mean_f1 + val_std_f1, color='green', alpha=0.2)

plt.title('Validation Curve: F1 Score vs. Distance Metrics (k=3) with SMOTE')
plt.xlabel('Distance Metrics')
plt.ylabel('F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


######## Learning curve KNN ########
# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize lists to store F1 scores
train_sizes = np.arange(0.1, 1.1, 0.1)  # Training sizes from 10% to 100%
train_f1_scores = []
test_f1_scores = []

# Best model parameters (already identified)
best_k = 3
best_metric = 'manhattan'
best_weight = 'uniform'

# Initialize SMOTE
smote = SMOTE(random_state=seed)

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

    # Apply SMOTE to the training data
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_partial, Y_train_partial)

    # Initialize KNN model with the best parameters
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weight)

    # Train the model on the resampled training set
    knn.fit(X_train_resampled, Y_train_resampled)

    # Predict on the training set
    Y_train_pred = knn.predict(X_train_resampled)

    # Calculate F1 score for the training set
    train_f1 = f1_score(Y_train_resampled, Y_train_pred, average='macro')

    # Append the results for the training set
    train_f1_scores.append(train_f1)
    
    if train_size < 1.0:
        # Predict on the test set only if it's not the full training set
        Y_test_pred = knn.predict(X_test_partial)

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

# Plotting the learning curve for F1 score
plt.figure(figsize=(8, 6))

# F1 Score curve with specified colors
plt.plot(train_sizes * 100, train_f1_scores, label='Training F1 Score', marker='o', color='blue')  # Blue for training
plt.plot([size * 100 for size in train_sizes if size < 1.0], 
         [f1 for f1 in test_f1_scores if f1 is not None], 
         label='Validation F1 Score', marker='o', color='green')  # Green for validation

plt.xlabel('Training Size (%)')
plt.ylabel('Weighted F1 Score')
plt.title('Learning Curve - Weighted F1 Score with SMOTE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

######## KNN without SMOTE ########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize lists to store results
results = []

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # Different distance metrics to evaluate
weights_options = ['uniform', 'distance']  # Weighting options for neighbors

# Loop through different k values, distance metrics, and weightings
for metric in metrics:  # Loop through each distance metric
    for weight in weights_options:  # Loop through each weighting option
        for k in k_values:  # Loop through each k value
            
            # Initialize KNN model with current k value, distance metric, and weighting
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)

            # Step 2b: Perform cross-validation using 10 folds
            kf = KFold(n_splits=10, shuffle=True, random_state=seed)
            
            # Store weighted F1 scores for each fold
            fold_f1_scores = cross_val_score(knn, X_train, Y_train, cv=kf, scoring='f1_macro')

            # Calculate and store the mean weighted F1 score across 10 folds
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

# Find the best combination of k, distance metric, and weighting
best_result = max(results, key=lambda x: x[3])  # Sort by mean F1 score (index 3)
best_k, best_metric, best_weight, best_f1_score, best_classification_report = best_result

# Display the best k, metric, weight, and F1 score
print(f"\nBest k: {best_k}, Best Metric: {best_metric}, Best Weight: {best_weight} with Mean F1 Score: {best_f1_score:.4f}")

# Print final results
print(f"\nClassification Report for Best k={best_k}, Metric={best_metric}, Weight={best_weight}:\n{best_classification_report}")

############ KNN WITH SMOTE - Grid Search for Comparison #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize lists to store results
results = []

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # Different distance metrics to evaluate
weights_options = ['uniform', 'distance']  # Weighting options for neighbors

# Define macro F1 as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Loop through different k values, distance metrics, and weightings
for metric in metrics:
    for weight in weights_options:
        for k in k_values:
            
            # Create a pipeline with SMOTE and KNN
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=seed)),
                ('knn', KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight))
            ])

            # Perform cross-validation using 10 folds on the training data
            fold_f1_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring=f1_scorer)

            # Calculate and store the mean F1 score across 10 folds
            mean_f1_score = np.mean(fold_f1_scores)

            # Append the result tuple (k, metric, weighting, mean F1 score)
            results.append((k, metric, weight, mean_f1_score))

            # Output progress
            print(f"k={k}, metric={metric}, weight={weight}: Mean F1 Score={mean_f1_score:.4f}")

# Find the best combination of k, distance metric, and weighting based on cross-validation F1 score
best_result_smote = max(results, key=lambda x: x[3])  # Sort by mean F1 score (index 3)
best_k_smote, best_metric_smote, best_weight_smote, best_f1_score_smote = best_result_smote

# Train the best model on the full training dataset and evaluate on the test set
best_pipeline = Pipeline([
    ('smote', SMOTE(random_state=seed)),
    ('knn', KNeighborsClassifier(n_neighbors=best_k_smote, metric=best_metric_smote, weights=best_weight_smote))
])

# Fit the best model on the full training set
best_pipeline.fit(X_train, Y_train)

# Predict on the test set
Y_test_pred = best_pipeline.predict(X_test)

# Display the best k, metric, weight, and F1 score
print(f"\nBest k: {best_k_smote}, Best Metric: {best_metric_smote}, Best Weight: {best_weight_smote} with Mean F1 Score: {best_f1_score_smote:.4f}")

# Output final classification report for the best model on the test set
final_classification_report = classification_report(Y_test, Y_test_pred)
print(f"\nClassification Report for Best Model on Test Set:\n{final_classification_report}")


########### Best KNN Model #########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Set the model parameters
k = 3
metric = 'manhattan'
weight = 'uniform'

# Create a pipeline with SMOTE and KNN (k=3, manhattan distance, uniform weights)
pipeline = Pipeline([
    ('smote', SMOTE(random_state=seed)),
    ('knn', KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight))
])

# Perform 10-fold cross-validation on the training data
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Evaluate cross-validation performance using F1 score with macro average
cv_f1_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring='f1_macro')
mean_cv_f1_score = cv_f1_scores.mean()

# Output the mean F1 score from 10-fold cross-validation
print(f"Mean F1 Macro Score from 10-Fold Cross-Validation: {mean_cv_f1_score:.4f}")

# Train the model on the full training set
pipeline.fit(X_train, Y_train)

# Predict on the training set
Y_train_pred = pipeline.predict(X_train)

# Calculate and output the F1 macro score on the full training set
train_f1_macro = f1_score(Y_train, Y_train_pred, average='macro')
print(f"F1 Macro Score on Full Training Set: {train_f1_macro:.4f}")

# Predict on the test set
Y_test_pred = pipeline.predict(X_test)

# Calculate and output the F1 macro score on the test set
test_f1_macro = f1_score(Y_test, Y_test_pred, average='macro')
print(f"F1 Macro Score on Test Set: {test_f1_macro:.4f}")


########## Validation Curve: Accuracy for different values of C. SVM ###########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define C values to evaluate
C_values = [0.001, 0.01, 0.1, 1, 10]
kernel_choice = 'linear'

# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Initialize lists to store F1 scores and standard deviations
train_mean_f1_scores = []
train_std_f1_scores = []
val_mean_f1_scores = []
val_std_f1_scores = []

# Create a pipeline with SMOTE and SVM
pipeline = Pipeline([
    ('smote', SMOTE(random_state=seed)),
    ('svm', SVC(kernel=kernel_choice, random_state=seed))
])

# Perform validation curve for SVM with SMOTE and varying C values
train_scores, val_scores = validation_curve(
    pipeline,
    X_train, Y_train,
    param_name='svm__C', param_range=C_values,  # Apply C values to the SVM step in the pipeline
    cv=10, scoring=f1_scorer
)

# Calculate the mean and standard deviation for training and validation F1 scores
train_mean_f1_scores = np.mean(train_scores, axis=1)
train_std_f1_scores = np.std(train_scores, axis=1)
val_mean_f1_scores = np.mean(val_scores, axis=1)
val_std_f1_scores = np.std(val_scores, axis=1)

# Plot the final graph comparing F1 scores for different C values (RBF kernel)
plt.figure(figsize=(10, 6))

# Plot training F1 scores with confidence bands
plt.plot(C_values, train_mean_f1_scores, label='Training F1 Score', marker='o', linestyle='solid', color='blue')
plt.fill_between(C_values, 
                 train_mean_f1_scores - train_std_f1_scores,
                 train_mean_f1_scores + train_std_f1_scores,
                 color='blue', alpha=0.2)

# Plot validation F1 scores with confidence bands
plt.plot(C_values, val_mean_f1_scores, label='Validation F1 Score', marker='o', linestyle='solid', color='green')
plt.fill_between(C_values, 
                 val_mean_f1_scores - val_std_f1_scores,
                 val_mean_f1_scores + val_std_f1_scores,
                 color='green', alpha=0.2)

# Customize the plot
plt.title('Comparison of F1 Scores for Linear Kernel with Different C Values (with SMOTE)')
plt.xscale('log')  # Set x-axis to log scale since C values span several orders of magnitude
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Macro F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()



######## When using linear curve then the best C value is trivial.
######## When using rbf curve then the best C value is 10.

############ Validation Curve SVM with SMOTE ############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define kernel methods and fixed C value
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_value = 10

# Define F1 score as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Initialize lists to store F1 scores and standard deviations
train_mean_f1_scores = []
train_std_f1_scores = []
val_mean_f1_scores = []
val_std_f1_scores = []

# Perform validation curve for each kernel with a fixed C value
for kernel in kernels:
    # Use validation_curve to compute training and validation F1 scores
    train_scores, val_scores = validation_curve(
        SVC(kernel=kernel, C=C_value, random_state=seed),
        X_train, Y_train,
        param_name='C', param_range=[C_value],  # Fix C to the single value
        cv=10, scoring=f1_scorer
    )

    # Calculate the mean and standard deviation for training and validation F1 scores
    train_mean_f1 = np.mean(train_scores, axis=1)[0]  # Only one C value, pick first
    train_std_f1 = np.std(train_scores, axis=1)[0]
    val_mean_f1 = np.mean(val_scores, axis=1)[0]
    val_std_f1 = np.std(val_scores, axis=1)[0]

    # Append the mean and std values to the corresponding lists
    train_mean_f1_scores.append(train_mean_f1)
    train_std_f1_scores.append(train_std_f1)
    val_mean_f1_scores.append(val_mean_f1)
    val_std_f1_scores.append(val_std_f1)

# Plot the final graph comparing F1 scores across different kernels

plt.figure(figsize=(10, 6))

# Plot training F1 scores with confidence bands
plt.plot(kernels, train_mean_f1_scores, label='Training F1 Score', marker='o', linestyle='solid', color='blue')
plt.fill_between(kernels, 
                 np.array(train_mean_f1_scores) - np.array(train_std_f1_scores),
                 np.array(train_mean_f1_scores) + np.array(train_std_f1_scores),
                 color='blue', alpha=0.2)

# Plot validation F1 scores with confidence bands
plt.plot(kernels, val_mean_f1_scores, label='Validation F1 Score', marker='o', linestyle='solid', color='green')
plt.fill_between(kernels, 
                 np.array(val_mean_f1_scores) - np.array(val_std_f1_scores),
                 np.array(val_mean_f1_scores) + np.array(val_std_f1_scores),
                 color='green', alpha=0.2)

# Customize the plot
plt.title(f'Comparison of F1 Scores for Different Kernels (C={C_value})')
plt.xlabel('Kernel')
plt.ylabel('Weighted F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()

############ Learning curve SVM with SMOTE ############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the best model parameters
best_kernel = 'rbf' 
best_C = 10  

# Define different training sizes (from 10% to 100%)
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize lists to store F1 scores for train and validation sets
train_f1_scores = []
val_f1_scores = []

# Initialize SMOTE
smote = SMOTE(random_state=seed)

# Loop through different training sizes
for train_size in train_sizes:
    
    if train_size < 1.0:
        # Perform the train-validation split with the specified train size
        X_train_partial, X_val, Y_train_partial, Y_val = train_test_split(X_train, Y_train, train_size=train_size, random_state=seed)
    else:
        # For 100% training, use the entire training set and no validation set
        X_train_partial, Y_train_partial = X_train, Y_train
        X_val, Y_val = None, None  # No validation data when training size is 100%

    # Apply SMOTE to the training data
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_partial, Y_train_partial)
    
    # Initialize the SVM model with the best kernel and C value
    svm = SVC(kernel=best_kernel, C=best_C, random_state=seed)
    
    # Train the model on the resampled training set
    svm.fit(X_train_resampled, Y_train_resampled)

    # Make predictions on the training set and calculate F1 score (macro)
    Y_train_pred = svm.predict(X_train_resampled)
    train_f1 = f1_score(Y_train_resampled, Y_train_pred, average='macro')
    train_f1_scores.append(train_f1)
    
    # If there is validation data (i.e., training size is less than 100%)
    if train_size < 1.0:
        # Make predictions on the validation set and calculate F1 score (macro)
        Y_val_pred = svm.predict(X_val)
        val_f1 = f1_score(Y_val, Y_val_pred, average='macro')
        val_f1_scores.append(val_f1)
        print(f"Training Size: {train_size*100:.0f}%, Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")
    else:
        # Append None for validation F1 score when training size is 100%
        val_f1_scores.append(None)
        print(f"Training Size: {train_size*100:.0f}%, Train F1: {train_f1:.4f}, No Validation F1 for 100% Training Set")

# Plot the learning curve for F1 score

plt.figure(figsize=(8, 6))

# Plot F1 score curves
plt.plot(train_sizes, train_f1_scores, label='Training F1 Score', marker='o', linestyle='solid', color='blue')
plt.plot([size for size in train_sizes if size < 1.0], 
         [f1 for f1 in val_f1_scores if f1 is not None], 
         label='Validation F1 Score', marker='o', linestyle='solid', color='green')

# Customize the plot
plt.title('Learning Curve: Macro F1 Score as a Function of Training Size (with SMOTE)')
plt.xlabel('Training Size')
plt.ylabel('Macro F1 Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()



############ SVM WITH SMOTE #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize vectors of different kernel methods and different values for C
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.001, 0.01, 0.1, 1, 10]

# Create KFold cross-validation object
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Initialize lists to store model results and classification reports
model_results = []  # To store (kernel, C, mean_f1_score)
class_reports_svm_smote = {}  # To store classification report for each model

# Define F1 (macro) as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Perform cross-validation for each combination of kernel and C
for kernel in kernels:
    for C in C_values:
        # Initialize the SVM model with the current kernel and C value
        svm = SVC(kernel=kernel, C=C, random_state=seed)

        # Create a pipeline to apply SMOTE followed by the SVM classifier
        smote = SMOTE(random_state=seed)
        pipeline = Pipeline([('smote', smote), ('svm', svm)])

        # Perform 10-fold cross-validation using the KFold object on training data
        cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring=f1_scorer)

        # Store the mean F1 score across 10 folds for training data evaluation
        mean_f1_score = np.mean(cv_scores)
        model_results.append((kernel, C, mean_f1_score))  # Store kernel, C, and F1 score

        # Train the pipeline on the full training set
        pipeline.fit(X_train, Y_train)

        # Predict on the test set
        Y_pred = pipeline.predict(X_test)

        # Store the classification report for each kernel and C combination in a dictionary
        # Remove the average='macro' argument from classification_report
        report = classification_report(Y_test, Y_pred)
        class_reports_svm_smote[(kernel, C)] = report
        
        # Output the classification report for the current model
        print(f"Kernel: {kernel}, C: {C}\n{report}")

        # Output progress
        print(f"Kernel: {kernel}, C: {C}, Mean F1 Score: {mean_f1_score:.4f}")

# Output the stored model results (kernel, C, and corresponding F1 score)
print("\nAll Model Results (Kernel, C, Mean F1 Score):")
for result in model_results:
    print(f"Kernel: {result[0]}, C: {result[1]}, Mean F1 Score: {result[2]:.4f}")

# Find the best model based on the highest mean F1 score
best_model_svm_smote = max(model_results, key=lambda x: x[2])
best_kernel, best_C, best_f1_score = best_model_svm_smote

# Output the best model and its classification report
print(f"\nBest Model -> Kernel: {best_kernel}, C: {best_C}, F1 Score: {best_f1_score:.4f}")
print("\nClassification Report for the Best Model:\n")
print(class_reports_svm_smote[(best_kernel, best_C)])

############ SVM WITHOUT SMOTE #############
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize vectors of different kernel methods and different values for C
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.01, 0.1, 1, 10]

# Create KFold cross-validation object
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Initialize lists to store model results
model_results = []  # To store (kernel, C, mean_f1_score)
classification_reports = {}  # To store classification reports for each model

# Perform cross-validation for each combination of kernel and C
for kernel in kernels:
    for C in C_values:
        # Initialize the SVM model with the current kernel and C value
        svm = SVC(kernel=kernel, C=C, random_state=seed)

        # Perform 10-fold cross-validation using the KFold object
        cv_scores = cross_val_score(svm, X_train, Y_train, cv=kf, scoring='f1_macro')

        # Store the mean F1 score across 10 folds
        mean_f1_score = np.mean(cv_scores)
        model_results.append((kernel, C, mean_f1_score))  # Store kernel, C, and F1 score
        
        # Train the model on the entire training set
        svm.fit(X_train, Y_train)

        # Predict on the test set
        Y_pred = svm.predict(X_test)
        
        # Store the classification report for each kernel and C combination
        report = classification_report(Y_test, Y_pred)  # Removed 'average' argument
        classification_reports[(kernel, C)] = report  # Store the report in the dictionary
        print(classification_reports[(kernel, C)])

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

# Best Model -> Kernel: rbf, C: 10, F1 Score: 0.9473
# Classification Report for the Best Model:

# >>> print(classification_reports[(best_kernel, best_C)])
#               precision    recall  f1-score   support

#            0       0.94      0.99      0.96       520
#            1       0.92      0.69      0.79       110

#     accuracy                           0.93       630
#    macro avg       0.93      0.84      0.87       630
# weighted avg       0.93      0.93      0.93       630

########## Best SVM Model #########
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize the SVM model with RBF kernel and C=10
svm = SVC(kernel='rbf', C=10, random_state=seed)

# Create KFold cross-validation object
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# Create a pipeline to apply SMOTE followed by the SVM classifier
smote = SMOTE(random_state=seed)
pipeline = Pipeline([('smote', smote), ('svm', svm)])

# Define F1 (macro) as the scoring metric
f1_scorer = make_scorer(f1_score, average='macro')

# Perform 10-fold cross-validation using the KFold object on training data
cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=kf, scoring=f1_scorer)

# Store the mean F1 score across 10 folds for training data evaluation
mean_f1_score_train_cv = np.mean(cv_scores)

# Train the pipeline on the full training set (after cross-validation)
pipeline.fit(X_train, Y_train)

# Predict on the full training set
Y_train_pred = pipeline.predict(X_train)

# Predict on the test set
Y_test_pred = pipeline.predict(X_test)

# Calculate the macro F1 score on the full training set
f1_score_train = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate the macro F1 score on the test set
f1_score_test = f1_score(Y_test, Y_test_pred, average='macro')

# Store the classification report for the test set
report_test = classification_report(Y_test, Y_test_pred)

# Output the results
print(f"Kernel: rbf, C: 10\n")
print(f"Mean F1 Score (CV on Training Set): {mean_f1_score_train_cv:.4f}")
print(f"F1 Score (Full Training Set): {f1_score_train:.4f}")
print(f"F1 Score (Test Set): {f1_score_test:.4f}")
print(f"\nClassification Report (Test Set):\n{report_test}")

####### Neural Networks ########
##### Neural Network Validation Curve for Activation Function #####
# Assuming X_scaled and Y are already defined
# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Different activation functions to test
activation_functions = ['relu', 'logistic', 'tanh']

# Lists to store training and validation F1 scores for each activation function
train_f1_scores = []
val_f1_scores = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Define macro F1 scorer
f1_scorer = make_scorer(f1_score, average='macro')

# Iterate over different activation functions
for activation in activation_functions:
    # Define the MLPClassifier with varying activation functions
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation=activation, solver='adam', random_state=seed, max_iter=500)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Validation curve for the current activation function
    train_score, val_score = validation_curve(pipeline, X_train, Y_train, param_name='mlpclassifier__activation',
                                              param_range=[activation], cv=kf, scoring=f1_scorer)
    
    # Store the mean F1 scores for training and validation sets
    train_f1_scores.append(np.mean(train_score))
    val_f1_scores.append(np.mean(val_score))

# Plotting the validation curve for each activation function
plt.figure(figsize=(10, 6))
plt.plot(activation_functions, train_f1_scores, label="Training F1 Score", marker='o', linestyle='solid', color='blue')
plt.plot(activation_functions, val_f1_scores, label="Validation F1 Score", marker='o', linestyle='solid', color='green')
plt.title("Validation Curve - Activation Functions (Macro F1 Score)")
plt.xlabel("Activation Function")
plt.ylabel("Macro F1 Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Validation Curve for Learning Rate in Neural Network with SMOTE ##########
# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Learning rates to test
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# Lists to store training and validation F1 scores for each learning rate
train_f1_scores = []
val_f1_scores = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Define macro F1 scorer
f1_scorer = make_scorer(f1_score, average='macro')

# Iterate over different learning rates
for learning_rate in learning_rates:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', random_state=seed, 
                          max_iter=500, learning_rate_init=learning_rate)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Validation curve for the current learning rate
    train_score, val_score = validation_curve(pipeline, X_train, Y_train, param_name='mlpclassifier__learning_rate_init',
                                              param_range=[learning_rate], cv=kf, scoring=f1_scorer)
    
    # Store the mean F1 scores for training and validation sets
    train_f1_scores.append(np.mean(train_score))
    val_f1_scores.append(np.mean(val_score))

# Plotting the validation curve for each learning rate
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, train_f1_scores, label="Training F1 Score", marker='o', linestyle='solid', color='blue')
plt.plot(learning_rates, val_f1_scores, label="Validation F1 Score", marker='o', linestyle='solid', color='green')
plt.title("Validation Curve - Learning Rates (Macro F1 Score, Tanh Activation)")
plt.xlabel("Learning Rate")
plt.ylabel("Macro F1 Score")
plt.xscale('log')  # Use logarithmic scale for learning rates
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

#### 0.001 learning rate is the best for this dataset. ####
#### Interesting that both adam and sgd solvers use 0.001 as the default learning rate. ####

######## Validation Curve for Batch Size in Neural Network with SMOTE ##########
# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the SMOTE object
smote = SMOTE(random_state=seed)

# Batch sizes to test
batch_sizes = [32, 64, 128, 256, 512]

# Lists to store training and validation F1 scores for each batch size
train_f1_scores = []
val_f1_scores = []

# Define the k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Define macro F1 scorer
f1_scorer = make_scorer(f1_score, average='macro')

# Iterate over different batch sizes
for batch_size in batch_sizes:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam', random_state=seed, 
                          max_iter=500, batch_size=batch_size, learning_rate_init=0.001)
    
    # Create a pipeline with SMOTE and the model
    pipeline = make_pipeline(smote, model)
    
    # Validation curve for the current batch size
    train_score, val_score = validation_curve(pipeline, X_train, Y_train, param_name='mlpclassifier__batch_size',
                                              param_range=[batch_size], cv=kf, scoring=f1_scorer)
    
    # Store the mean F1 scores for training and validation sets
    train_f1_scores.append(np.mean(train_score))
    val_f1_scores.append(np.mean(val_score))

# Plotting the validation curve for each batch size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, train_f1_scores, label="Training F1 Score", marker='o', linestyle='dashed', color='blue')
plt.plot(batch_sizes, val_f1_scores, label="Validation F1 Score", marker='o', linestyle='solid', color='green')
plt.title("Validation Curve - Batch Sizes (Macro F1 Score, ReLU Activation, Learning Rate 0.001)")
plt.xlabel("Batch Size")
plt.ylabel("Macro F1 Score")
plt.xscale('log')  # Use logarithmic scale for batch sizes
plt.xticks(batch_sizes, labels=batch_sizes)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# 64 batch size performed best.

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
    model = MLPClassifier(hidden_layer_sizes=layer_config, activation='tanh', solver='adam', batch_size=64,
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
        
        # Calculate F1 scores (macro) for both partial training and validation sets
        train_f1 = f1_score(Y_train_partial, Y_train_pred, average='macro')
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='macro')
        
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

# 64, 32, 16 performed best.

### Learning curve for Neural Network with SMOTE ###
# Split the data into training and test (test will not be used in this case)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Store results for plotting
train_f1_scores = []
val_f1_scores = []
train_sizes = np.arange(0.1, 1.1, 0.1)  # Train sizes from 10% to 100%

# Loop over different train-test splits
for train_size in train_sizes:
    # Check if the training size is 100% (train_size == 1.0)
    if train_size == 1.0:
        # Train on the full dataset without splitting for validation
        X_train_partial, Y_train_partial = X_train, Y_train
        X_val_partial, Y_val_partial = None, None  # No validation set
    else:
        # Split the train dataset into train_partial and validation_partial with varying train size
        X_train_partial, X_val_partial, Y_train_partial, Y_val_partial = train_test_split(
            X_train, Y_train, train_size=train_size, random_state=seed)
    
    # Initialize SMOTE and MLPClassifier
    smote = SMOTE(random_state=seed)
    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='adam', alpha=0.001, batch_size=64, random_state=seed, max_iter=500)
    
    # Combine SMOTE and the classifier in a pipeline
    pipeline = make_pipeline(smote, model)
    
    # Train the model on the partial training set
    pipeline.fit(X_train_partial, Y_train_partial)
    
    # Predict on the training partial set
    Y_train_pred = pipeline.predict(X_train_partial)
    
    # Calculate F1 score on the training partial set (macro F1)
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='macro')
    
    # Store the training F1 score
    train_f1_scores.append(train_f1)
    
    if train_size == 1.0:
        # No validation score if training size is 100%
        val_f1_scores.append(None)
    else:
        # Predict on the validation partial set and calculate the validation F1 score (macro F1)
        Y_val_pred = pipeline.predict(X_val_partial)
        val_f1 = f1_score(Y_val_partial, Y_val_pred, average='macro')
        val_f1_scores.append(val_f1)

    # Output progress
    print(f"Train Size: {train_size*100:.0f}%, Train F1: {train_f1:.4f}, Validation F1: {val_f1 if val_f1 is not None else 'N/A'}")

# Plot Learning Curves: F1 Score
plt.figure(figsize=(12, 6))

# Plot training F1 scores
plt.plot(train_sizes * 100, train_f1_scores, label="Training F1 Score", marker='o', linestyle='solid', color='blue')

# Plot validation F1 scores, excluding the 100% training size
plt.plot(train_sizes[:-1] * 100, [f1 for f1 in val_f1_scores if f1 is not None], label="Validation F1 Score", marker='o', linestyle='solid', color='green')

plt.title("Learning Curve - Macro F1 Score vs. Training Size")
plt.xlabel("Training Size (%)")
plt.ylabel("Macro F1 Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

######## Learning Curve for Neural Network with SMOTE ##########
# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Initialize SMOTE and set up neural network with warm_start for incremental learning
smote = SMOTE(random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='adam', random_state=seed, batch_size=64,alpha=0.001, max_iter=1,early_stopping=True, warm_start=True)

# Combine SMOTE and the classifier in a pipeline
pipeline = make_pipeline(smote, model)

# Lists to store F1 scores over epochs
train_f1_scores = []
val_f1_scores = []

# Define the number of epochs (iterations)
n_epochs = 500

# Split the training set into train_partial and validation_partial
X_train_partial, X_val_partial, Y_train_partial, Y_val_partial = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed)

# Train the model for n_epochs and record performance at each epoch
for epoch in range(n_epochs):
    # Set max_iter to 1 for warm start and incremental training
    model.set_params(max_iter=epoch + 1)  # Incremental epochs
    
    # Train on the partial training set
    pipeline.fit(X_train_partial, Y_train_partial)

    # Predict on the training partial set
    Y_train_pred = pipeline.predict(X_train_partial)
    
    # Predict on the validation partial set
    Y_val_pred = pipeline.predict(X_val_partial)

    # Calculate F1 scores on the training and validation partial sets
    train_f1 = f1_score(Y_train_partial, Y_train_pred, average='macro')
    val_f1 = f1_score(Y_val_partial, Y_val_pred, average='macro')
    
    # Store the F1 scores for plotting
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)

    # Output progress
    print(f"Epoch: {epoch+1}, Train F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")

# Find the epoch with the highest validation F1 score
best_epoch = np.argmax(val_f1_scores) + 1
best_val_f1 = max(val_f1_scores)

# Plot Learning Curve: F1 Score
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_epochs + 1), train_f1_scores, label="Training F1 Score", marker='o', linestyle='-', color='blue')
plt.plot(range(1, n_epochs + 1), val_f1_scores, label="Validation F1 Score", marker='o', linestyle='-', color='green')

# Mark the epoch with the highest validation F1 score
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch} (F1: {best_val_f1:.4f})')

plt.title("Learning Curve - Macro F1 Score vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Macro F1 Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output the epoch with the highest validation F1 score
print(f"\nEpoch with the highest Validation F1 Score: {best_epoch}, Validation F1 Score: {best_val_f1:.4f}")



######### Neural Network Model Evaluation ##########
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', random_state=seed, 
                      max_iter=500, batch_size=32, alpha = 0.01, learning_rate_init=0.001)

model.fit(X_train, Y_train)
Y_test_pred = model.predict(X_test)

# Predict on the training set for training performance
Y_train_pred = model.predict(X_train)

train_f1 = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate F1 score on the test data
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

print(f"Training F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# with default alpha (weight decay) of 0.0001
# Training score 0.9714
# Test Score 0.8954

# alpha 0.1 -> 0.9455, 0.8983
# alpha 0.01 -> 0.9497, 0.9015

######## Neural Network with CV #######
# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define KFold cross-validation with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Define the neural network model (MLPClassifier) with specified batch size, learning rate, and hidden layers
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', random_state=seed, 
                      max_iter=500, batch_size=32, learning_rate_init=0.001)

# Perform cross-validation and get predictions for each fold on the training data
Y_train_pred = cross_val_predict(model, X_train, Y_train, cv=kf)

# Train the model on the full training set and evaluate on the test set
model.fit(X_train, Y_train)
Y_test_pred = model.predict(X_test)

# Calculate F1 score on cross-validated training data
train_f1 = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate F1 score on the test data
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

# Print performance metrics for training and test sets
print(f"Cross-validated Training F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# alpha 0.01
# Training F1 Score: 0.9280 
# Test F1 Score: 0.9015

# default alpha 0.001
# Training F1 Score: 0.9208
# Test F1 Score: 0.8954

### Note that the performance when using cross validation reduced for the training data set.
### This is likely due to cross validation improving the generalization of the model.

#### Neural network with CV and SMOTE ####
# Initialize SMOTE and MLPClassifier with batch size, learning rate, and hidden layers
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
smote = SMOTE(random_state=seed)
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='adam', random_state=seed, 
                      max_iter=500, batch_size=64, alpha=0.001, learning_rate_init=0.001)

# Combine SMOTE and the classifier in a pipeline
pipeline = make_pipeline(smote, model)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Perform cross-validation with SMOTE applied to the training set in each fold
Y_train_pred_cv = cross_val_predict(pipeline, X_train, Y_train, cv=kf)

# Train the model on the full training set and evaluate on the test set
pipeline.fit(X_train, Y_train)

# Predict on the full training set
Y_train_pred_full = pipeline.predict(X_train)

# Predict on the test set
Y_test_pred = pipeline.predict(X_test)

# Calculate F1 score on the cross-validated training data
train_f1_cv = f1_score(Y_train, Y_train_pred_cv, average='macro')

# Calculate F1 score on the full training set
train_f1_full = f1_score(Y_train, Y_train_pred_full, average='macro')

# Calculate F1 score on the test data
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

# Print performance metrics for cross-validated training, full training, and test sets
print(f"Cross-validated Training F1 Score: {train_f1_cv:.4f}")
print(f"Full Training F1 Score: {train_f1_full:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# relu & batch 32 & default alpha 0.001
# Training F1 Score: 0.9133
# Test F1 Score: 0.9032

# relu & batch 32 & alpha 0.01
# Training F1 Score: 0.9143
# Test F1 Score: 0.9155

# tanh & batch 32 & alpha 0.001
# Training F1 Score: 0.9264
# Test F1 Score: 0.9260

# tanh & batch 32 & alpha 0.01
# Training F1 Score: 0.9240
# Test F1 Score: 0.9072

# tanh & batch 64 & alpha 0.001
# Training F1 Score: 0.9258
# Test F1 Score: 0.9260

# tanh & batch 64 & alpha 0.0001
# Training F1 Score: 0.9263
# Test F1 Score: 0.9166

### Reduced the training accuracy and F1 score but improved the test accuracy and F1 score.

# Grid search to evaluate the reasonableness of my model performance. Cross check.
# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (64, 32), (64, 32, 16)],  # Different layer configurations
    'learning_rate_init': [0.001, 0.01, 0.1],  # Different learning rates
    'batch_size': [32, 64],  # Different batch sizes
    'activation': ['relu', 'tanh'],  # Activation functions to try
    'alpha': [0.001, 0.01]  # Regularization parameter (L2 penalty)
}

# Create the GridSearchCV object
# Set 'scoring' to 'f1_macro' to focus on F1 score, and use 5-fold cross-validation (cv=5)
grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=500, random_state=seed), 
                           param_grid=param_grid, cv=5, scoring='f1_macro')

# Fit the model with grid search on training data
grid_search.fit(X_train, Y_train)

# Best hyperparameters found by grid search
print("Best Hyperparameters: ", grid_search.best_params_)

# Predict using the best model on the training data
Y_train_pred = grid_search.predict(X_train)

# Predict using the best model on the test data
Y_test_pred = grid_search.predict(X_test)

# Calculate F1 score on the training data
train_f1 = f1_score(Y_train, Y_train_pred, average='macro')

# Calculate F1 score on the test data
test_f1 = f1_score(Y_test, Y_test_pred, average='macro')

# Print the F1 scores for the training and test sets
print(f"Training F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Grid Search Results
# Best Hyperparameters:  {'activation': 'tanh', 'alpha': 0.001, 'batch_size': 64,
# 'hidden_layer_sizes': (64, 32, 16), 'learning_rate_init': 0.001}
# Training F1 Score: 0.9711
# Test F1 Score: 0.9113

######## Measuring Training Time (Clock time and iterations) for SVM ########
# Measure wall clock time
# Best SVM model parameters
# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
best_kernel = 'rbf'
best_C = 10

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
best_k = 3
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
hidden_layers = (64, 32, 16)
activation_fn = 'tanh'
best_solver = 'adam'
best_batch_size = 64
best_alpha = 0.001
best_learning_rate = 0.001
max_iter = 500

# Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Measure wall clock time for training the neural network
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_fn, solver=best_solver, random_state=seed, 
                      max_iter=500, batch_size=best_batch_size, alpha=best_alpha, learning_rate_init=best_learning_rate)
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

# print(f"SVM Number of iterations: {svc_linear.n_iter_}")
print(f"Neural Network (MLPClassifier) Iterations: {mlp_iterations}")

print(f"SVM Training Time (Wall Clock): {time_svm_training} seconds")
print(f"KNN Training Time (Wall Clock): {time_knn_training} seconds")
print(f"Neural Network (MLPClassifier) Training Time: {time_mlp_training} seconds")

print(f"SVM Prediction Time (Wall Clock): {time_svm_prediction} seconds")
print(f"KNN Prediction Time (Wall Clock): {time_knn_prediction} seconds")
print(f"Neural Network (MLPClassifier) Prediction Time: {time_mlp_prediction} seconds")

# KNN Results
# F1 Score on Full Training Set: 0.9582
# F1 Score on Test Set: 0.8809

#SVM Results
# F1 Score (Full Training Set): 0.9099
# F1 Score (Test Set): 0.8661

# Neural Network Results
# Full Training F1 Score: 0.9705
# Test F1 Score: 0.9260

##### End of Iranian Drug Project #####










