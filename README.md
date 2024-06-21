# ML-Classifier-Project

The project utilizes Random Forest or Multi-Layered-Perceptron models to predict classes for two datasets: one for diabetes (either type 1, type 2 or none) and one for a loan approval status (approved or declined).

## How it works

We start by reading the datasets from the csv files. We already have the split between training and testing data, but we first have to analyze the data and preprocess it.

Initially, we analyze the numerical and categorical/discrete data and create histograms and bar plots for each of them. We will also plot the class distribution for the target variable in order to see if the dataset is balanced or not.

We continue by **analyzing the correlation** between the numerical features (using the **Pearson criteria**) and between the categorical features (using the **chi-squared test**) in order to see if there are any **features that are highly correlated and can be removed**. We will also plot some heatmaps to visualize the correlation.

Note: All the heatmaps/histograms/bar plots will be saved as images in their respective folders.

We follow-up by **imputing the missing values** in the dataset, using the mean for numerical features and the most frequent value for categorical features. After this, we will also **impute the outliers** in the dataset, using the **interquartile range method** and then **standardize the data** using the StandardScaler. **One-hot encoding will be used for the categorical features**.

These will be passed through a preprocessing pipeline before being fed to the models.

We are running two models: first one is a ***Random Forest Classifier*** - for which we have two implementations (one using the RandomForestClassifier from _sklearn.ensemble_ and one using an implementation from scratch done for an university lab -  for which we will need to convert the numerical features to categorical ones using a discretization method) and the second one is a ***Multi-Layered Perceptron Classifier*** (using the MLPClassifier from _sklearn.neural_network_).

We evaluate the models using the accuracy score, the confusion matrix and the classification report.

Recommened to run the code in a virtual environment. To install the required packages, run the following commands in the project directory:
```
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas matplotlib seaborn graphviz ipython
```

