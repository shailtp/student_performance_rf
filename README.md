# Student Performance Prediction using Random Forest Classifier

This project predicts student performance (pass/fail) using a **Random Forest Classifier** based on features such as **Previous Grades**, **Study Hours per Week**, **Attendance Rate**, and more. The model was trained and tested using the **Cleaned Student Performance Dataset**, and detailed feature importance was also analyzed.

## Dataset

The dataset used for this project can be accessed from [Kaggle: Student Performance Prediction Dataset](https://www.kaggle.com/datasets/souradippal/student-performance-prediction/data).

## Summary

This project implements a machine learning pipeline to preprocess the dataset, build a Random Forest model, and evaluate its performance. It also includes feature ranking using the Gini Index to identify the most important factors contributing to student success.

## Steps

1. **Load and Preprocess Data**
   - Load the dataset and clean any unnecessary or missing data.
   - Convert categorical features (e.g., Participation in Extracurricular Activities) into numeric values.
   - Remove irrelevant columns (e.g., unnamed columns, student IDs).

2. **Split Data**
   - Separate the features and target variable (`Passed`).
   - Split the data into training (80%) and testing (20%) subsets.

3. **Hyperparameter Tuning**
   - Use GridSearchCV to find the best combination of hyperparameters for the Random Forest Classifier.
   - Tune key parameters such as the number of trees (`n_estimators`) and the number of features considered at each split (`max_features`).

4. **Train and Evaluate Model**
   - Train the Random Forest model on the training dataset using the best parameters.
   - Evaluate the model using metrics such as Accuracy, Precision, Recall, and F1 Score.
   - Display and plot a confusion matrix for better insight into the model's predictions.

5. **Feature Ranking**
   - Compute feature importance using the Gini Index from the trained Random Forest model.
   - Display a ranked list of features and visualize their importance in a bar chart.

6. **Train and Test with Top 3 Features**
   - Use only the top 3 most important features to train a new Random Forest model.
   - Test the model on the testing dataset and evaluate its performance using the same metrics.
   - Display and plot the confusion matrix for this reduced feature set.

7. **Predict Probabilities for Specific Samples**
   - Select one positive and one negative sample from the test dataset.
   - Use the trained model to predict probabilities for these samples.
   - Display the predicted class, probabilities for each class, and whether the prediction was correct.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student_performance_rf.git
   cd student_performance_rf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook student_performance.ipynb
   ```

4. Follow the steps in the notebook to reproduce the results or test new datasets.
