import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Callable, List
from graphviz import Digraph, Source
from IPython.display import display as idisplay
from copy import deepcopy
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

diabetes_train = pd.read_csv('tema2_Diabet/Diabet_train.csv')
diabetes_test = pd.read_csv('tema2_Diabet/Diabet_test.csv')
credit_train = pd.read_csv('tema2_Credit_Risk/credit_risk_train.csv')
credit_test = pd.read_csv('tema2_Credit_Risk/credit_risk_test.csv')
diabetes_full = pd.read_csv('tema2_Diabet/Diabet_full.csv')
credit_full = pd.read_csv('tema2_Credit_Risk/credit_risk_full.csv')

output_dir_diabetes = 'class_distribution/diabetes'
os.makedirs(output_dir_diabetes, exist_ok=True)
output_dir_credit = 'class_distribution/credit'
os.makedirs(output_dir_credit, exist_ok=True)
output_dir_hist_diabetes = 'histograms/diabetes'
os.makedirs(output_dir_hist_diabetes, exist_ok=True)
output_dir_hist_credit = 'histograms/credit'
os.makedirs(output_dir_hist_credit, exist_ok=True)

output_dir_corr = 'correlation_analysis'
os.makedirs(output_dir_corr, exist_ok=True)

continuous_numeric_diabetes = [
    'psychological-rating', 'BodyMassIndex', 'Age', 
    'CognitionScore', 'Body_Stats', 'Metabolical_Rate'
]
discrete_diabetes = [
    'HealthcareInterest', 'PreCVA', 'RoutineChecks', 'alcoholAbuse', 
    'cholesterol_ver', 'vegetables', 'HighBP', 'Unprocessed_fructose', 
    'Jogging', 'IncreasedChol', 'gender', 'myocardial_infarction', 
    'Cardio', 'ImprovedAveragePulmonaryCapacity', 'Smoker'
]
ordinal_diabetes = ['CompletedEduLvl', 'HealthScore', 'SalaryBraket']

continuous_numeric_credit = [
    'applicant_age', 'applicant_income', 'job_tenure_years', 
    'loan_amount', 'loan_rate', 'loan_income_ratio', 
    'credit_history_length_years', 'credit_history_length_months'
]
discrete_credit = [
    'residential_status', 'loan_purpose', 'loan_rating', 
    'credit_history_default_status', 'stability_rating'
]
ordinal_credit = [] 


# Function to analyze numeric attributes
def analyze_numeric(df, attributes, dataset_name):
    numeric_attributes = df[attributes]
    summary = numeric_attributes.describe(percentiles=[.25, .5, .75]).T
    summary['missing_values'] = df.shape[0] - numeric_attributes.count()
    summary.rename(columns={
        '50%': 'median',
        '25%': '25%',
        '75%': '75%'
    }, inplace=True)
    print(summary)

    # Boxplot for numeric attributes
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=numeric_attributes)
    plt.xticks(rotation=90)
    plt.title(f'Boxplot of Continuous Numeric Attributes - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_hist_diabetes if dataset_name == 'Diabetes' else output_dir_hist_credit, f'numeric_boxplot_{dataset_name}.png'))
    plt.close()
    return summary

def analyze_categorical(df, attributes, dataset_name, ordinal_columns):
    summary_data = []

    for col in attributes:
        num_unique_values = df[col].nunique()
        num_missing_values = df[col].isna().sum()
        num_samples_no_missing = df.shape[0] - num_missing_values

        attribute_type = 'Ordinal' if col in ordinal_columns else 'Discrete'
        
        summary_data.append({
            'Attribute': col,
            'Type': attribute_type,
            'Number of unique values': num_unique_values,
            'Number of samples with no missing values': num_samples_no_missing,
            'Number of missing values': num_missing_values
        })

        # Print detailed information
        print(f'Attribute: {col} (Type: {attribute_type})')
        print(f'Number of unique values: {num_unique_values}')
        print(f'Number of missing values: {num_missing_values}')
        print(df[col].value_counts())
        print()

        # Save histogram for categorical attributes
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col])
        plt.xticks(rotation=90)
        plt.title(f'Distribution of {col} ({attribute_type}) - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_hist_diabetes if dataset_name == 'Diabetes' else output_dir_hist_credit, f'{col}_distribution_{dataset_name}.png'))
        plt.close()

    # Create summary DataFrame and display it
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    # Save the summary table to a CSV file
    summary_df.to_csv(os.path.join(output_dir_hist_diabetes if dataset_name == 'Diabetes' else output_dir_hist_credit, f'categorical_summary_{dataset_name}.csv'), index=False)
    return summary_df

def plot_class_distribution(df, target_col, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title(f'Class Distribution in {dataset_name} Dataset')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_diabetes if 'Diabetes' in dataset_name else output_dir_credit, f'{dataset_name}_class_distribution.png'))
    plt.show()


# Function to analyze correlation between numeric attributes
def analyze_numeric_correlation(df, attributes, dataset_name):
    print(f'Correlation Matrix for {dataset_name} Dataset')
    print()
    numeric_attributes = df[attributes]
    corr_matrix = numeric_attributes.corr(method='pearson')
    print(corr_matrix)

    # Visualizing the correlation matrix with seaborn heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
    plt.title(f'Correlation Matrix for {dataset_name} Dataset', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_corr, f'correlation_matrix_{dataset_name}.png'))
    plt.close()

    return corr_matrix

# Function to analyze correlation between categorical attributes using Chi-Square test
def analyze_categorical_correlation(df, attributes, dataset_name):
    print(f'P-Values Matrix for Categorical Attributes in {dataset_name} Dataset')
    print()
    categorical_attributes = df[attributes]
    p_values = pd.DataFrame(index=attributes, columns=attributes)

    for col1 in categorical_attributes.columns:
        for col2 in categorical_attributes.columns:
            if col1 == col2:
                p_values.loc[col1, col2] = 1
                continue
            
            attribute1 = categorical_attributes[col1]
            attribute2 = categorical_attributes[col2]
            contingency_table = pd.crosstab(attribute1, attribute2)
            _, p, _, _ = chi2_contingency(contingency_table)
            p_values.loc[col1, col2] = 1-p
            p_values.loc[col2, col1] = 1-p
    
    # Print the p-values matrix
    print(p_values)

    # Save the p-values matrix
    p_values.to_csv(os.path.join(output_dir_corr, f'categorical_correlation_{dataset_name}.csv'))

    return p_values

#------------------------------------------------------------------------------------------------

# Analyze numeric and categorical attributes for diabetes dataset
diabetes_numeric_summary = analyze_numeric(diabetes_full, continuous_numeric_diabetes, 'Diabetes')
diabetes_categorical_summary = analyze_categorical(diabetes_full, discrete_diabetes + ordinal_diabetes, 'Diabetes', ordinal_diabetes)

# Analyze numeric and categorical attributes for credit risk dataset
credit_numeric_summary = analyze_numeric(credit_full, continuous_numeric_credit, 'Credit')
credit_categorical_summary = analyze_categorical(credit_full, discrete_credit + ordinal_credit, 'Credit', ordinal_credit)

#------------------------------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print()

# Plot class distribution for the diabetes training and test datasets
plot_class_distribution(diabetes_train, 'Diabetes', 'Diabetes_Train')
plot_class_distribution(diabetes_test, 'Diabetes', 'Diabetes_Test')

# Plot class distribution for the credit risk training and test datasets
plot_class_distribution(credit_train, 'loan_approval_status', 'Credit_Train')
plot_class_distribution(credit_test, 'loan_approval_status', 'Credit_Test')


#------------------------------------------------------------------------------------------------
# Analyze correlation for numeric attributes in diabetes dataset
print()
print("-----------------------------------------------------------------------")
print()
numeric_corr_matrix_diabetes = analyze_numeric_correlation(diabetes_full, continuous_numeric_diabetes, 'Diabetes')

# Analyze correlation for categorical attributes in diabetes dataset
categorical_p_values_diabetes = analyze_categorical_correlation(diabetes_full, discrete_diabetes + ordinal_diabetes, 'Diabetes')

# Analyze correlation for numeric attributes in credit risk dataset
numeric_corr_matrix_credit = analyze_numeric_correlation(credit_full, continuous_numeric_credit, 'Credit')

# Analyze correlation for categorical attributes in credit risk dataset
categorical_p_values_credit = analyze_categorical_correlation(credit_full, discrete_credit + ordinal_credit, 'Credit')



#------------------------------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print()

# Identify missing data in diabetes dataset

# Identify missing data in diabetes dataset
missing_data_diabetes = diabetes_full.isna().sum()
missing_data_diabetes = missing_data_diabetes[missing_data_diabetes > 0]
print("Missing data in diabetes dataset:")
print(missing_data_diabetes)

# Identify missing data in credit risk dataset
missing_data_credit = credit_full.isna().sum()
missing_data_credit = missing_data_credit[missing_data_credit > 0]
print("Missing data in credit risk dataset:")
print(missing_data_credit)

# Function to apply univariate imputation separately for numeric and non-numeric data
def univariate_imputation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    imputer_mean = SimpleImputer(strategy='mean')
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')

    df_numeric_imputed = pd.DataFrame(imputer_mean.fit_transform(df[numeric_cols]), columns=numeric_cols)
    df_non_numeric_imputed = pd.DataFrame(imputer_most_frequent.fit_transform(df[non_numeric_cols]), columns=non_numeric_cols)

    imputed_df = pd.concat([df_numeric_imputed, df_non_numeric_imputed], axis=1)
    return imputed_df

# Apply univariate imputation to diabetes dataset
diabetes_full_imputed = univariate_imputation(diabetes_full)

# Apply univariate imputation to credit risk dataset
credit_full_imputed = univariate_imputation(credit_full)

# Function to apply multivariate imputation only to numeric data
def multivariate_imputation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    imputer = IterativeImputer(max_iter=10, random_state=0)
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # Combine numeric and non-numeric columns back
    imputed_df = pd.concat([df_numeric_imputed, df[non_numeric_cols].reset_index(drop=True)], axis=1)
    return imputed_df

# Apply multivariate imputation to diabetes dataset
diabetes_full_multivariate = multivariate_imputation(diabetes_full_imputed)

# Apply multivariate imputation to credit risk dataset
credit_full_multivariate = multivariate_imputation(credit_full_imputed)

# Save the imputed datasets for diabetes
diabetes_full_imputed.to_csv('diabetes_full_imputed.csv', index=False)
diabetes_full_multivariate.to_csv('diabetes_full_multivariate.csv', index=False)

# Save the imputed datasets for credit risk
credit_full_imputed.to_csv('credit_full_imputed.csv', index=False)
credit_full_multivariate.to_csv('credit_full_multivariate.csv', index=False)

#------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------")

print()

# Function to identify and impute extreme values
def impute_extreme_values(df, numerical_attributes):
    imputer = SimpleImputer(strategy='mean') 
    
    for col in numerical_attributes:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        extreme_values = ((df[col] < lower_bound) | (df[col] > upper_bound))
        
        print(f"{col}: {extreme_values.sum()} extreme values")

        # Replace extreme values with np.nan
        df.loc[extreme_values, col] = np.nan
        
        
    df[numerical_attributes] = imputer.fit_transform(df[numerical_attributes])
    
    return df

# Apply extreme value imputation to diabetes dataset
diabetes_full_extreme_imputed = impute_extreme_values(diabetes_full, continuous_numeric_diabetes)
print()

# Apply extreme value imputation to credit risk dataset
credit_full_extreme_imputed = impute_extreme_values(credit_full, continuous_numeric_credit)

# Save the imputed datasets for diabetes
diabetes_full_extreme_imputed.to_csv('diabetes_full_extreme_imputed.csv', index=False)

# Save the imputed datasets for credit risk
credit_full_extreme_imputed.to_csv('credit_full_extreme_imputed.csv', index=False)

#------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------")

print()


# List of attributes to remove
# Basically, we look for the attributes with very high correlation, and we remove the one with the
# highest number of high correlations with other attributes
credit_risk_list_remove = [
    'stability_rating', 
    'loan_rating', 
    'credit_history_default_status', 
    'credit_history_length_months', 
    'applicant_age', 
    'residential_status'
]
diabetes_list_remove = [
    'HighBP', 
    'Jogging', 
    'PreCVA', 
    'gender', 
    'myocardial_infarction', 
    'Smoker', 
    'ImprovedAveragePulmonaryCapacity', 
    'HealthScore', 
    'CompletedEduLvl', 
    'vegetables', 
    'Unprocessed_fructose', 
    'HealthcareInterest', 
    'SalaryBraket', 
    'IncreasedChol', 
    'cholesterol_ver', 
    'Cardio',
    'Metabolical_Rate'
]

remaining_numeric_credit = [col for col in continuous_numeric_credit if col not in credit_risk_list_remove]
remaining_numeric_diabetes = [col for col in continuous_numeric_diabetes if col not in diabetes_list_remove]
remaining_attributes_diabetes = [col for col in diabetes_full.columns if col not in diabetes_list_remove]
remaining_attributes_credit = [col for col in credit_full.columns if col not in credit_risk_list_remove]

# Remove specified attributes from the datasets
credit_full_reduced = credit_full.drop(columns=credit_risk_list_remove)
diabetes_full_reduced = diabetes_full.drop(columns=diabetes_list_remove)

# Save the reduced datasets to new CSV files
credit_full_reduced.to_csv('credit_full_reduced.csv', index=False)
diabetes_full_reduced.to_csv('diabetes_full_reduced.csv', index=False)


#------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------")

print()

# Function to standardize numerical attributes
def standardize_numerical_attributes(df, numerical_attributes):
    scaler = StandardScaler()
    df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])
    return df


# Standardize numerical attributes in both datasets
diabetes_full_standardized = standardize_numerical_attributes(diabetes_full, continuous_numeric_diabetes)
credit_full_standardized = standardize_numerical_attributes(credit_full, continuous_numeric_credit)

# Save the standardized datasets to new CSV files
diabetes_full_standardized.to_csv('diabetes_full_standardized.csv', index=False)
credit_full_standardized.to_csv('credit_full_standardized.csv', index=False)


#------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------")

print()


print("Random Forest with sklearn")
print()

diabetes_train_reduced = diabetes_train.drop(columns=diabetes_list_remove)
diabetes_test_reduced = diabetes_test.drop(columns=diabetes_list_remove)
credit_train_reduced = credit_train.drop(columns=credit_risk_list_remove)
credit_test_reduced = credit_test.drop(columns=credit_risk_list_remove)

# Define target attributes
target_diabetes = 'Diabetes'
target_credit = 'loan_approval_status'

# Separate features and target variables
X_train_diabetes = diabetes_train_reduced.drop(columns=[target_diabetes])
y_train_diabetes = diabetes_train_reduced[target_diabetes]
X_test_diabetes = diabetes_test_reduced.drop(columns=[target_diabetes])
y_test_diabetes = diabetes_test_reduced[target_diabetes]

X_train_credit = credit_train_reduced.drop(columns=[target_credit])
y_train_credit = credit_train_reduced[target_credit]
X_test_credit = credit_test_reduced.drop(columns=[target_credit])
y_test_credit = credit_test_reduced[target_credit]

#remove nan values using imputer 
X_train_diabetes = multivariate_imputation(X_train_diabetes)
X_test_diabetes = multivariate_imputation(X_test_diabetes)

#remove extreme values
X_train_diabetes = impute_extreme_values(X_train_diabetes, remaining_numeric_diabetes)
X_test_diabetes = impute_extreme_values(X_test_diabetes, remaining_numeric_diabetes)


#remove nan values using imputer function univariate
X_train_credit = univariate_imputation(X_train_credit)
X_test_credit = univariate_imputation(X_test_credit)

#remove extreme values
X_train_credit = impute_extreme_values(X_train_credit, remaining_numeric_credit)
X_test_credit = impute_extreme_values(X_test_credit, remaining_numeric_credit)

# Identify categorical features for credit dataset
categorical_features_credit = X_train_credit.select_dtypes(include=['object']).columns.tolist()

# Standardize numerical attributes for diabetes dataset
X_train_diabetes = standardize_numerical_attributes(X_train_diabetes, remaining_numeric_diabetes)
X_test_diabetes = standardize_numerical_attributes(X_test_diabetes, remaining_numeric_diabetes)

# Standardize numerical attributes for credit dataset
X_train_credit = standardize_numerical_attributes(X_train_credit, remaining_numeric_credit)
X_test_credit = standardize_numerical_attributes(X_test_credit, remaining_numeric_credit)

# Preprocess credit dataset - one hot encoding for categorical features and standardization for numerical features
preprocessor_credit = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), remaining_numeric_credit),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_credit)
    ],
    remainder='passthrough'
)

# Fit the preprocessor on the training data and transform both the training and test data
#make a copy of the X_train_credit and X_test_credit first
X_train_credit_copy = X_train_credit.copy()
X_test_credit_copy = X_test_credit.copy()

X_train_credit_copy = preprocessor_credit.fit_transform(X_train_credit)
X_test_credit_copy = preprocessor_credit.transform(X_test_credit)

# Function to train and evaluate a RandomForest model
def train_evaluate_random_forest(X_train, X_test, y_train, y_test, dataset_name):
    rf = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print(f"Results for {dataset_name} Dataset")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print()

# Train and evaluate the RandomForest model on both datasets
train_evaluate_random_forest(X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes, "Diabetes")
train_evaluate_random_forest(X_train_credit_copy, X_test_credit_copy, y_train_credit, y_test_credit, "Credit Risk")
#------------------------------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("Random Forest with an implementation from scratch")
print()

class DecisionTreeNode:
    """
    Un nod din arborele de decizie. Acesta poate fi un nod intermediar sau un nod frunză.
    """
    
    def __init__(self, 
                 feature: Optional[str] = None, 
                 children: Optional[Dict[str, 'DecisionTreeNode']] = None,
                 label: Optional[str] = None):
        """
        Constructor pentru un nod din arborele de decizie
        
        Args:
            feature (str, optional): 
                Numele atributului după care se face împărțirea. Defaults to None.
            children (Dict[str, DecisionTreeNode], optional): 
                Un dictionar ce conține subarborii nodului curent. Defaults to None.
            label (str, optional): 
                Clasa nodului frunză. Defaults to None.
        """
        self.split_feature = feature  # Numele atributului după care se face împărțirea (None pentru nodurile frunză)
        self.children = children if (children is not None and feature is not None) else {}
        self.label = label    # Clasa nodului frunză (None pentru nodurile intermediare)
        self.depth = 1        # Adâncimea nodului în arbore (se calculează în timpul construcției arborelui)
        self.score = 0        # Scorul nodului (se calculează în timpul construcției arborelui)
        self.num_samples = 0  # Numărul de exemple din setul de date care ajung în nodul curent
    
    def get_tree_graph(self,
                       graph: Digraph = None) -> Digraph:
        """
        Construiește reprezentarea grafică a arborelui de decizie folosind biblioteca Graphviz
    
        Args:
            graph (Digraph, optional): 
                Obiectul Digraph în care se construiește reprezentarea arborelui. Defaults to None.
        """
        if graph is None:
            graph = Digraph()
            graph.attr('node', shape='box')
    
        if self.split_feature is None:
            # Nod frunză
            graph.node(f"{self}", f"Label: {self.label}\n"
                                  f"Score: {self.score:.3f}\n"
                                  f"Samples: {self.num_samples}", 
                       fillcolor='darkolivegreen2', style='filled')
        else:
            # Nod intermediar
            graph.node(f"{self}", f"Split: {self.split_feature}?\n"
                                  f"Score: {self.score:.3f}\n"
                                  f"Samples: {self.num_samples}", fillcolor='lightblue', style='filled')
            
            for value, child in self.children.items():
                child.get_tree_graph(graph)
                graph.edge(f"{self}", f"{child}", label=f"{value}")
    
        return graph
    
    def display(self):
        """
        Afișează arborele de decizie folosind biblioteca Graphviz. Arborele va fi afișat ca output al celulei.
        """
        graph = self.get_tree_graph()
        idisplay(Source(graph.source))


class DecisionTree:
    """
    Clasa care implementează un arbore de decizie. 
    Arborele poate fi construit folosind algoritmul ID3 sau Random Tree, în funcție de strategia de împărțire specificată.
    """
    def __init__(self,
                 split_strategy: str = 'random',
                 max_depth: int = np.inf,
                 min_samples_per_node: int = 1):
        """
        Constructor pentru un arbore de decizie
        
        Args:
            split_strategy (string, optional): 
                Strategia folosită pentru alegerea împărțirii într-un nod. Aceasta poate fi:
                - 'id3' - alege împărțirea care maximizează câștigul informațional (folosind algoritmul ID3)
                - 'random' - alege aleator o împărțire
                Defaults to 'random'.
            max_depth (int, optional): 
                Adâncimea maximă a arborelui. Defaults to infinity.
            min_samples_per_node (int, optional): 
                Numărul minim de exemple dintr-un nod pentru a face o împărțire. 
                Defaults to 1.
        """
        self._root: DecisionTreeNode | None = None # Rădăcina arborelui
        self._split_strategy: str = split_strategy
        self._max_depth: int = max_depth
        self._min_samples_per_node: int = min_samples_per_node
        
        
    @staticmethod
    def most_frequent_class(y: pd.Series) -> str:
        """
        Obține clasa majoritară din setul de date
        
        Args:
            y (pd.DataFrame): 
                Vectorul de clase. Fiecare element reprezintă clasa unui exemplu din setul de date
        
        Returns:
            str: 
                Clasa majoritară din setul de date
        
        """
      
        return y.mode().values[0]
            
    
    @staticmethod
    def compute_entropy(y: pd.Series) -> float:
        """
        Calculează entropia setului de date
        
        Args:
            y (pd.Series): 
                Vectorul de clase. Fiecare element reprezintă clasa unui exemplu din setul de date
        
        Returns:
            float: 
                Entropia setului de date
        
        """

        freq = y.value_counts(sort=False)
        entropy = 0
        for val in freq:
            entropy += -(val/len(y)) * np.log2(val/len(y))
        return entropy
        
    
    @staticmethod
    def compute_information_gain(X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Calculează câștigul informațional al unui atribut din setul de date
        
        Args:
            X (pd.DataFrame): 
                Setul de date (atributele)
            y (pd.Series): 
                Clasele corespunzătoare fiecărui exemplu din setul de date
            feature (str): 
                Numele atributului pentru care se calculează câștigul informațional
        
        Returns:
            float: 
                Câștigul informațional al atributului
        """
   
        initial_entropy = DecisionTree.compute_entropy(y)
        final_entropy = 0
        for value in X[feature].unique():
            final_entropy += (len(X[X[feature] == value]) / len(X)) * DecisionTree.compute_entropy(y[X[feature] == value])
        return initial_entropy - final_entropy
    
    
    def _select_random_split_feature(self, X: pd.DataFrame, y: pd.Series, attribute_list: list[str]) -> str:
        """
        Alege în mod aleator atributul după care se face împărțirea într-un nod
        
        Args:
            X (pd.DataFrame): 
                Setul de date (atributele)
            y (pd.Series): 
                Clasele corespunzătoare fiecărui exemplu din setul de date
            attribute_list (list[str]): 
                Lista de atribute rămase pentru construcția arborelui
        
        Returns:
            str: 
                Numele atributului după care se face împărțirea
                
        """
        
        #   Pentru a alege un element aleator dintr-o listă puteți folosi funcția np.random.choice()
        return np.random.choice(attribute_list)

    
    
    def _select_best_split_feature(self, X: pd.DataFrame, y: pd.Series, attribute_list: list[str]) -> str:
        max_gain = -np.inf
        best_feature = attribute_list[0]  # Default to the first attribute in the list
        for feature in attribute_list:
            gain = DecisionTree.compute_information_gain(X, y, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature
    
    
    def _generate_tree(self,
                       parent_node: DecisionTreeNode | None,
                       X: pd.DataFrame,
                       y: pd.Series,
                       feature_list: list[str],
                       select_feature_func: Callable[[pd.DataFrame, pd.Series, list[str]], str]) -> DecisionTreeNode:
        """
        Construiește arborele de decizie pe baza setului de date X și a claselor țintă y
        
        Args:
            parent_node (DecisionTreeNode): 
                Nodul părinte al nodului curent
            X (pd.DataFrame): 
                Setul de date (atributele)
            y (pd.Series): 
                Clasele corespunzătoare fiecărui exemplu din setul de date
            feature_list (list[str]): 
                Lista de atribute rămase pentru construcția arborelui
            select_feature_func (Callable[[pd.DataFrame, pd.Series, list[str]], str]):
                Funcția folosită pentru a alege atributul după care se face împărțirea
                
        Returns:
            DecisionTreeNode: 
                Nodul rădăcină al arborelui de decizie construit
                
        """
        # Se face o copie a listei de atribute pentru a nu modifica lista inițială
        feature_list = deepcopy(feature_list)
        
        # Se creează un nou nod pentru arbore
        node = DecisionTreeNode()
        node.depth = parent_node.depth + 1 if parent_node is not None else 0
        node.score = DecisionTree.compute_entropy(y)  
        node.num_samples = len(y)
        node.label = DecisionTree.most_frequent_class(y)
        
     
        if y.nunique() == 1 or node.depth >= self._max_depth or len(y) < self._min_samples_per_node or len(feature_list) == 0:
            return node
        
     
        split_feature = select_feature_func(X, y, feature_list)
        feature_list.remove(split_feature)
        node.split_feature = split_feature
        for value in X[split_feature].unique():
            node.children[value] = self._generate_tree(node, X[X[split_feature] == value], y[X[split_feature] == value], feature_list, select_feature_func)
        return node
    
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Construiește arborele de decizie pe baza setului de date. 
        Va folosi strategia de împărțire specificată în constructor.
        
        Args:
            X (pd.DataFrame): 
                Setul de date (atributele)
            y (pd.Series): 
                Clasele corespunzătoare fiecărui exemplu din setul de date
        """
        # Selectează funcția de împărțire a nodurilor
        if self._split_strategy == 'random':
            select_feature_func = self._select_random_split_feature
        elif self._split_strategy == 'id3':
            select_feature_func = self._select_best_split_feature
        else:
            raise ValueError(f"Unknown split strategy {self._split_strategy}")
        
        self._root = self._generate_tree(parent_node=None,
                                         X=X,
                                         y=y,
                                         feature_list=X.columns.tolist(),
                                         select_feature_func=select_feature_func)
        
    def _predict_once(self, x: pd.Series) -> str:
        """
        Realizează predicția clasei pentru un singur exemplu x
        
        Args:
            x (pd.Series): 
                Atributele unui exemplu din setul de date
        
        Returns:
            str: 
                Clasa prezisă pentru exemplul x
                
        """
        node = self._root
        
        while node.split_feature is not None:
            if node.split_feature in x and x[node.split_feature] in node.children:
                node = node.children[x[node.split_feature]]
            else:
                break
        return node.label
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realizează predicția claselor pentru un set de date X
        
        Args:
            X (pd.DataFrame): Setul de date (atributele) pentru care se dorește clasificarea

        Returns:
            np.ndarray: Un vector cu clasele prezise pentru fiecare exemplu din X
            
        """
        return np.array([self._predict_once(x) for _, x in X.iterrows()])
    
    def get_depth(self) -> int:
        """
        Returnează adâncimea arborelui
        
        Returns:
            int: Adâncimea arborelui
        """
        # Se parcurge arborele pentru a găsi adâncimea maximă
        def _get_depth(node: DecisionTreeNode) -> int:
            if node is None:
                return 0
            return max([_get_depth(child) for child in node.children.values()], default=0) + 1
        
        return _get_depth(self._root)
    
    def get_number_of_nodes(self) -> int:
        """
        Returnează numărul de noduri din arbore
        
        Returns:
            int: Numărul de noduri din arbore
        """
        # Se parcurge arborele pentru a găsi numărul de noduri
        def _get_number_of_nodes(node: DecisionTreeNode) -> int:
            if node is None:
                return 0
            return sum([_get_number_of_nodes(child) for child in node.children.values()], 0) + 1
        
        return _get_number_of_nodes(self._root)
    
    def get_tree_graph(self) -> Digraph:
        """
        Construiește reprezentarea grafică a arborelui de decizie folosind biblioteca Graphviz
        
        Returns:
            Digraph: Obiectul Digraph în care se construiește reprezentarea arborelui
        """
        return self._root.get_tree_graph()
    
    def display(self):
        """
        Afișează arborele de decizie folosind biblioteca Graphviz. Arborele va fi afișat ca output al celulei.
        """
        return self._root.display()
    

class RandomForest:
    """
    Clasa care implementează un clasificator de tip pădure de arbori aleatori.
    """
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_per_node: int = 1,
                 split_strategy: str = 'random',
                 subset_size_ratio: float = 0.5,
                 subset_feature_ratio: float = 0.75):
        """
        Constructor pentru un clasificator de tip pădure de arbori aleatori
        
        Args:
            n_estimators (int, optional): 
                Numărul de arbori din pădure. Defaults to 100.
            max_depth (int, optional): 
                Adâncimea maximă a fiecărui arbore. Defaults to 3.
            min_samples_per_node (int, optional): 
                Numărul minim de exemple dintr-un nod pentru a face o împărțire. 
                Defaults to 1.
            split_strategy (str, optional):
                Strategia folosită pentru alegerea împărțirii într-un nod. Aceasta poate fi:
                - 'id3' - alege împărțirea care maximizează câștigul informațional (folosind algoritmul ID3)
                - 'random' - alege aleator o împărțire
                Defaults to 'random'.
            subset_size_ratio (float, optional):
                Raportul de dimensiune al subsetului de date folosit pentru construirea fiecărui arbore comparativ cu
                dimensiunea setului de date inițial. Trebuie să fie un număr între 0 și 1.
                Defaults to 0.5.
            subset_feature_ratio (float, optional):
                Raportul de dimensiune al subsetului de atribute folosit pentru construirea fiecărui arbore comparativ cu
                dimensiunea setului de atribute inițial. Trebuie să fie un număr între 0 și 1.
                Defaults to 0.75.
        """
        assert 0 < subset_size_ratio <= 1, "subset_size_ratio must be between 0 and 1"
        assert 0 < subset_feature_ratio <= 1, "subset_feature_ratio must be between 0 and 1"
        
        self._trees: list[DecisionTree] = []
        self._n_estimators: int = n_estimators
        self._max_depth: int = max_depth
        self._min_samples_per_node: int = min_samples_per_node
        self._split_strategy: str = split_strategy
        self._subset_size_ratio: float = subset_size_ratio
        self._subset_feature_ratio: float = subset_feature_ratio
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Construiește pădurea de arbori aleatori pe baza setului de date
        
        Args:
            X (pd.DataFrame): 
                Setul de date (atributele)
            y (pd.Series): 
                Clasele corespunzătoare fiecărui exemplu din setul de date
        """
 
        n_samples = int(self._subset_size_ratio * X.shape[0])
        n_features = int(self._subset_feature_ratio * len(X.columns))

        for _ in range(self._n_estimators):
            indices = np.random.choice(X.index, size=n_samples, replace=False)
            features = np.random.choice(X.columns, size=n_features, replace=False)

            X_subset = X.loc[indices, features]
            y_subset = y.loc[indices]

            tree = DecisionTree(
                split_strategy=self._split_strategy,
                max_depth=self._max_depth,
                min_samples_per_node=self._min_samples_per_node
            )
            tree.fit(X_subset, y_subset)

            self._trees.append(tree)
        

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realizează predicția claselor pentru un set de date X
        
        Args:
            X (pd.DataFrame): Setul de date (atributele) pentru care se dorește clasificarea
            
        Returns:
            np.ndarray: Un vector cu clasele prezise pentru fiecare exemplu din X
        """
        predictions = []
        
        for tree in self._trees:
            predictions.append(tree.predict(X))
            
        # Se alege clasa majoritară pentru fiecare exemplu din setul de date
        return np.array([Counter(pred).most_common(1)[0][0] for pred in np.array(predictions).T])
    
    def display(self, max_trees: int = 5):
        """
        Afișează arborii din pădure
        
        Args:
            max_trees (int, optional): 
                Numărul maxim de arbori care vor fi afișați. Defaults to 5.
        
        Warnings:
            Afișarea arborilor nu este indicată pentru un număr mare de estimatori
        """
        for i, tree in enumerate(self._trees[:max_trees]):
            print()
            tree.display()

# Train and evaluate the RandomForest model on diabetes dataset
random_forest_diabetes = RandomForest(n_estimators=100, max_depth=10, split_strategy='id3')

def convert_numerical_to_categorical(df: pd.DataFrame, n_bins: int = 5, strategy: str = 'uniform') -> pd.DataFrame:
    df = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df[numerical_cols] = discretizer.fit_transform(df[numerical_cols])
    return df

# Convert the numerical attributes in the datasets

X_train_credit_c = univariate_imputation(X_train_credit)
X_test_credit_c = univariate_imputation(X_test_credit)

X_train_diabetes_c = multivariate_imputation(X_train_diabetes)
X_test_diabetes_c = multivariate_imputation(X_test_diabetes)

#remove extreme values
X_train_diabetes_c = impute_extreme_values(X_train_diabetes_c, remaining_numeric_diabetes)
X_test_diabetes_c = impute_extreme_values(X_test_diabetes_c, remaining_numeric_diabetes)


X_train_diabetes_c = convert_numerical_to_categorical(X_train_diabetes)
X_test_diabetes_c = convert_numerical_to_categorical(X_test_diabetes)
X_train_credit_c = convert_numerical_to_categorical(X_train_credit_c)
X_test_credit_c = convert_numerical_to_categorical(X_test_credit_c)

random_forest_diabetes.fit(X_train_diabetes_c, y_train_diabetes)
y_pred_diabetes = random_forest_diabetes.predict(X_test_diabetes_c)

print("Diabetes Dataset")
print("Accuracy:", accuracy_score(y_test_diabetes, y_pred_diabetes))
print("Classification Report:\n", classification_report(y_test_diabetes, y_pred_diabetes))
print("Confusion Matrix:\n", confusion_matrix(y_test_diabetes, y_pred_diabetes))
print()

# Train and evaluate the RandomForest model on credit risk dataset
random_forest_credit = RandomForest(n_estimators=100, max_depth=10, split_strategy='id3')
random_forest_credit.fit(X_train_credit_c, y_train_credit)
y_pred_credit = random_forest_credit.predict(X_test_credit_c)

print("Credit Risk Dataset")
print("Accuracy:", accuracy_score(y_test_credit, y_pred_credit))
print("Classification Report:\n", classification_report(y_test_credit, y_pred_credit))
print("Confusion Matrix:\n", confusion_matrix(y_test_credit, y_pred_credit))
print()


#------------------------------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print('MLP with sklearn')
print()


# Encode categorical variables
categorical_features_diabetes = X_train_diabetes.select_dtypes(include=['object']).columns.tolist()
categorical_features_credit = X_train_credit.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessing pipeline for credit dataset
preprocessor_credit = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in X_train_credit.columns if col not in categorical_features_credit]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_credit)
    ])

# Encode target variables
# label_encoder_diabetes = LabelEncoder()
# y_train_diabetes = label_encoder_diabetes.fit_transform(y_train_diabetes)
# y_test_diabetes = label_encoder_diabetes.transform(y_test_diabetes)

# label_encoder_credit = LabelEncoder()
# y_train_credit = label_encoder_credit.fit_transform(y_train_credit)
# y_test_credit = label_encoder_credit.transform(y_test_credit)


# Define the MLP model for diabetes dataset
mlp_diabetes = Pipeline(steps=[
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',  # Activation function
        solver='adam',  # Optimizer
        alpha=0.0001,  # L2 regularization (alpha is the L2 penalty (regularization term) parameter)
        learning_rate='adaptive',
        max_iter=200,  # Number of epochs
        random_state=0,
        early_stopping=True,  # Early stopping to prevent overfitting
        validation_fraction=0.1,  # 10% of training data for validation
        batch_size='auto'
    ))
])

# Define the MLP model for credit dataset
mlp_credit = Pipeline(steps=[
    ('preprocessor', preprocessor_credit),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',  # Activation function
        solver='adam',  # Optimizer
        alpha=0.0001,  # L2 regularization (alpha is the L2 penalty (regularization term) parameter)
        learning_rate='adaptive',
        max_iter=200,  # Number of epochs
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,  # 10% of training data for validation
        batch_size='auto'
    ))
])


# Train and evaluate the MLP model on diabetes dataset

mlp_diabetes.fit(X_train_diabetes, y_train_diabetes)
y_pred_diabetes = mlp_diabetes.predict(X_test_diabetes)

print("Diabetes Dataset")
print("Accuracy:", accuracy_score(y_test_diabetes, y_pred_diabetes))
print("Classification Report:\n", classification_report(y_test_diabetes, y_pred_diabetes))
print("Confusion Matrix:\n", confusion_matrix(y_test_diabetes, y_pred_diabetes))
print()

# Train and evaluate the MLP model on credit dataset


mlp_credit.fit(X_train_credit, y_train_credit)
y_pred_credit = mlp_credit.predict(X_test_credit)

print("Credit Risk Dataset")
print("Accuracy:", accuracy_score(y_test_credit, y_pred_credit))
print("Classification Report:\n", classification_report(y_test_credit, y_pred_credit))
print("Confusion Matrix:\n", confusion_matrix(y_test_credit, y_pred_credit))
print()