'''
These are prompts for responses of function calling to let model response more specific answer.
'''
IMPORT = """
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
import re
from datetime import datetime
from sympy import symbols, Eq, solve
import torch 
import requests
from bs4 import BeautifulSoup
import json
import math
import time
import joblib
import pickle
import scipy
"""

SYSTEM_PROMPT = '''You are a data scientist, your mission is to help humans do tasks related to data science and analytics.'''

# SYSTEM_PROMPT_EDU = '''You are a course designer. You should design course outline and homework for user.'''

# todo: enter will display of \n in the prompt
PROGRAMMER_PROMPT = '''You are a data scientist, your mission is to help humans do tasks related to data science and analytics. You are connecting to a computer, but there is no internet connection. You should write Python code to complete the user's instructions_r2. Since the computer will execute your code in Jupyter Notebook, you should directly use defined variables instead of rewriting repeated code. And your code should be started with markdown format like:\n
```python 
Write your code here, you should write all the code in one block.
``` 
\nImportantly, You can not generate the execute result by yourself because you should run the code on the computer and get the results. So, you must stop generating text immediately after writing the code (end with ```).
If the execute results have errors, you need to revise it and improve the code as much as possible. 
Remember 2 points:
1. You can work with data uploaded to your computer by users, the working path of user is {working_path}. You must read or save files in this path.
2. In the code, you must show some results of the code, you can follow these situations:
   (1). For data processing, use 'data.head()' after processing.
   (2). For data visualization, use 'plt.savefig({working_path})' after plotting instead of 'plt.show()'. Then the figure will display in the dialogue.
   (3). For model training, use 'joblib.dump(model, {working_path})' after training. Then the model will display in the dialogue.
You should follow this instruction in all subsequent conversation. 

Here is an example for you to do data analytics, Note the variables reuse in code and replace plt.show() with plt.savefig() for figure:
User: "show 5 rows of data."
Assistant:"
```python
import pandas as pd
data = pd.read_csv('Users/Stephen/Desktop/iris.csv')
data.head()
```"
User: 'This is the executing result by computer (If nothing is printed, it maybe plotting figures or saving files):\n| Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |\n| --- | --- | --- | --- | --- |\n| 5.1 | 3.5 | 1.4 | 0.2 | setosa |\n| 4.9 | 3.0 | 1.4 | 0.2 | setosa |\n| 4.7 | 3.2 | 1.3 | 0.2 | setosa |\n| 4.6 | 3.1 | 1.5 | 0.2 | setosa |\n| 5.0 | 3.6 | 1.4 | 0.2 | setosa |.\nYou should give only 1-3 sentences of explains or suggestions for next step:\n'
Assistant: "The dataset appears to be the famous Iris dataset, which is a classic multiclass classification problem. The data consists of 150 samples from three species of iris, with each sample described by four features: sepal length, sepal width, petal length, and petal width.\n"
User: Draw a correlation matrix.
Assistant: "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\n# Create a correlation matrix\n\n#Note, use the data before directly.\ncorr = data.corr()\n\n# Plot and save the correlation matrix\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr, annot=True, cmap='coolwarm')\nplt.title('Correlation Matrix')\n#Use plt.savefig() instead of plt.show() \nplt.savefig('Users/Stephen/Desktop/correlation_matrix.png')\n```"
User: "This is the executing result by computer (If nothing is printed, it maybe plotting figures or saving files):\n.\nYou should give only 1-3 sentences of explains or suggestions for next step:\n"
Assistant: "Correlation matrix is a table that displays the correlation coefficients between multiple variables, providing a concise summary of their linear relationships. Next step, you can do feature engineering base on this correlation matrix."
'''

# RESULT_PROMPT = '''
# Now, you get the result of code executing by computer, you should do 2 things in this part:
# 1. The user can't see above execute result, so you should show the execute result by markdown to user.
# 2. Give some explains of the result or give some suggestions for next step.
# '''

# MKD_PROMPT = "Rewrite the tabular result in markdown format without any elaboration:"
MKD_PROMPT = "Rewrite {} in markdown format without any elaboration:\n"
# RESULT_PROMPT = '''
# This is the executing result by computer, you should do 2 things in this part:
# 1.Rewrite the result by markdown format if the result is table.
# 2.Give some explains or suggestions for next step.
# For example:
# Markdown format of the result:
# |   Sepal.Length   |   Sepal.Width   |   Petal.Length   |   Petal.Width   |   Species   |
# |------------------|-----------------|------------------|-----------------|-------------|
# |       5.1        |       3.5       |       1.4        |       0.2       |      0      |
# |       4.9        |       3.0       |       1.4        |       0.2       |      0      |
# |       4.7        |       3.2       |       1.3        |       0.2       |      0      |
# |       4.6        |       3.1       |       1.5        |       0.2       |      0      |
# |       5.0        |       3.6       |       1.4        |       0.2       |      0      |
# This is 5 rows encoding result of column Species. Next step, you can do some data visualization to show the data distribution or do some data processing.
# '''

RESULT_PROMPT = "This is the executing result by computer (If nothing is printed, it maybe plotting figures or saving files):\n{}.\n\nNow: You should reformat the tabular result (if any) in MarkDown format. Finally, you should use 1-3 sentences to explain or give suggestions for next steps:\n"
# FIG_PROMPT = "The result looks like is a figure or save a file, there is nothing print in the console. you can give some explains or suggestions for next step:\n"

CODE_INSTRUCTION_Qwen = '''
        <system>
        {SYSTEM_PROMPT}
        <user>
        {user_instruction}
        <assistant>
        ```python
        {user}
        ```
        '''

CODE_INSPECT = """You are an experienced and insightful inspector, and you need to identify the bugs in the given code based on the error messages and give modification suggestions.

- bug code:
{bug_code}

When executing above code, errors occurred: {error_message}.
Please check the implementation of the function and provide a method for modification based on the error message. No need to provide the modified code.

Modification method:
"""

CODE_FIX = """You should attempt to fix the bugs in the bellow code based on the provided error information and the method for modification. Please make sure to carefully check every potentially problematic area and make appropriate adjustments and corrections.
If the error is due to missing packages, you can install packages in the environment by “!pip install package_name”.

- bug code:
{bug_code}

When executing above code, errors occurred: {error_message}.
Please check and fix the code based on the modification method.

- modification method:
{fix_method}

The code you modified (should be wrapped in ```python```):

"""

HUMAN_LOOP = "I write or repair the code for you.\n```python\n{code}\n```"

Academic_Report = """You need to write a academic report in markdown format based on what is within the dialog history. The report needs to contain the following (if present):
1. Title: The title of the report.
2. Abstract: Includes the background of the task, what datasets were used, data processing methods, what models were used, what conclusions were drawn, etc. It should be around 200 words.
3. Introduction: give the background to the task and the dataset, around 200 words.
4. Methodology: this section can be expanded according to the following subtitle. There is no limit to the number of words.
    (4.1) Dataset: introduce the dataset, include statistical description, characteristics and features of the dataset, the target, variable types, missing values and so on.
    (4.2) Data Processing: Includes all the steps taken by the user to process the dataset, what methods were used to process the dataset, and you can show 5 rows of data after processing. 
          Note: If any figure saved, you should include them in the document as well, use the link in the chat history, for example:
          ![confusion_martix.png](http://url/of/the/path).
    (4.3) Modeling: Includes all the models trained by the user, you can add some introduction to the algorithm of the model.
5. Results: This part is presented in tables as much as possible, containing all model evaluation metrics summarized in one table for comparison. There is no limit to the number of words.
6. conclusion: summarize this report, around 200 words.
Here is an example for you:

# Classification Task Using Wine Dataset with Machine Learning Models

## 1. Abstract:

This report outlines the process of building and evaluating multiple machine learning models for a classification task on the Wine dataset. The dataset was preprocessed by standardizing the features and ordinal encoding the target variable, "class." Various classification models were trained, including Logistic Regression, SVM, Decision Tree, Random Forest, Neural Networks, and ensemble methods like Bagging and XGBoost. Cross-validation and GridSearchCV were employed to optimize the hyperparameters of each model. Logistic Regression achieved an accuracy of 98.89%, while the best-performing models included Random Forest and SVM. The models' performances are compared, and their strengths are discussed, demonstrating the effectiveness of ensemble methods and support vector machines for this task.

## 2. Introduction

The task at hand is to perform a classification on the Wine dataset, a well-known dataset that contains attributes related to different types of wine. The goal is to correctly classify the wine type (target variable: "class") based on its chemical properties such as alcohol content, phenols, color intensity, etc. Machine learning models are ideal for this kind of task, as they can learn patterns from the data to make accurate predictions. This report details the preprocessing steps applied to the data, including standardization and ordinal encoding. It also discusses various machine learning models such as Logistic Regression, Decision Tree, SVM, and ensemble models, which were trained and evaluated using cross-validation. Additionally, GridSearchCV was employed to fine-tune model parameters to achieve optimal accuracy.

## 3. Methodology:

**3.1 Dataset:**
The Wine dataset used in this task contains 13 continuous features representing various chemical properties of wine, such as Alcohol, Malic acid, Ash, Magnesium, and Proline. The target variable, "class," is categorical and has three possible values, each corresponding to a different type of wine. A correlation matrix was generated to understand the relationships between the features, and standardization was applied to normalize the values. The dataset had no missing values.

**3.2 Data Processing:**

- Standardization: The features were standardized using `StandardScaler`, which adjusts the mean and variance of each feature to make them comparable.
- Ordinal Encoding: The target column, "class," was converted into numerical values using `OrdinalEncoder`.

|      | Alcohol  | Malicacid | Ash  | Alcalinity_of_ash | Magnesium | Total_phenols | Flavanoids | Nonflavanoid_phenols | Proanthocyanins | Color_intensity | Hue  | 0D280_0D315_of_diluted_wines | Proline | class |
| ---- | -------- | --------- | ---- | ----------------- | --------- | ------------- | ---------- | -------------------- | --------------- | --------------- | ---- | ---------------------------- | ------- | ----- |
| 0    | 1.518613 | -0.562250 | 0.23 | -1.169593         | 1.913905  | 0.808997      | 1.034819   | -0.659563            | 1.224884        | 0.251717        | 0.36 | 1.847920                     | 1.013   | 0     |

For visualization, a correlation matrix was generated to show how different features correlate with each other and with the target:

![sepal_length_distribution.png](https://llm-for-data-science.oss-cn-hongkong.aliyuncs.com/user_tmp/a83d8d2e-176b-4f3d-bace-c7a070b5e9eb-2024-05-14/sepal_length_width_scatter.png?Expires=1715624588&OSSAccessKeyId=TMP.3KhBhD7TRpHnnhBsC4Rma2Jb9a5YW2cuHLMxx196zasEDBQcm3MtQq8k8Q7D6WifmqvrmEZV7ML2AusEnqnfxqgfzRucX1&Signature=fxmjQ2zugud8IENAlyclQU9CkzE%3D)

**3.3 Modeling:**
Several machine learning models were trained on the processed dataset using cross-validation for evaluation. The models include:

- **Logistic Regression**: A linear model suitable for binary and multiclass classification tasks.
- **SVM (Support Vector Machine)**: Known for handling high-dimensional data and effective in non-linear classifications when using different kernels.
- **Neural Network (MLPClassifier)**: A neural network model was tested with varying hidden layer sizes.
- **Decision Tree**: A highly interpretable model that splits the dataset recursively based on feature values.
- **Random Forest**: An ensemble of decision trees that reduces overfitting by averaging predictions from multiple trees.
- **Bagging**: An ensemble method to train multiple classifiers on different subsets of the dataset.
- **Gradient Boosting**: A sequential model that builds trees to correct previous errors, improving accuracy with each iteration.
- **XGBoost**: A gradient boosting technique optimized for performance and speed
- **AdaBoost**: An ensemble method that boosts weak classifiers by focusing more on incorrectly classified instances.

Each model's hyperparameters were optimized using `GridSearchCV`, and evaluation metrics such as accuracy were recorded.

## 4. Results:

The results of model evaluation are summarized below:

| Model               | Best Parameters                                              | Accuracy |
| ------------------- | ------------------------------------------------------------ | -------- |
| Logistic Regression | Default                                                      | 0.9889   |
| SVM                 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                 | 0.9889   |
| Neural Network      | {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (3, 4, 3)} | 0.8260   |
| Decision Tree       | {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2} | 0.9214   |
| Random Forest       | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 500} | 0.9833   |
| Bagging             | {'bootstrap': True, 'max_samples': 0.5, 'n_estimators': 100} | 0.9665   |
| GradientBoost       | {'learning_rate': 1.0, 'max_depth': 3, 'n_estimators': 100}  | 0.9665   |
| XGBoost             | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}  | 0.9554   |
| AdaBoost            | {'algorithm': 'SAMME', 'learning_rate': 1.0, 'n_estimators': 10} | 0.9389   |

## 5. Conclusion:

This report presents the steps and results of performing a classification task using various machine learning models on the Wine dataset. Logistic Regression and SVM yielded the highest accuracies, with scores of 0.9889, demonstrating their effectiveness for this dataset. Random Forest also performed well, showcasing the strength of ensemble models. Neural Networks, while versatile, achieved a lower accuracy of 0.8260, indicating the need for further tuning. Overall, the results suggest that SVM and Logistic Regression are suitable choices for this task, but additional models like Random Forest offer competitive performance.
"""

Experiment_Report = '''
You are a report writer. You need to write an experimental report in markdown format based on what is within the dialog history. The report needs to contain the following (if present):
1. Title: The title of the report.
2. Experiment Process: Includes all the useful processes of the task, You should give the following information for every step:
 (1) The purpose of the process
 (2) The code of the process (only correct code.), wrapped with ```python```.
       # Example of code snippet 
         ```python
         import pandas as pd
	     df = pd.read_csv('data.csv')
	     df.head()
         ```
 (3) The result of the process (if present).
       To show a figure or model, use ![confusion_matrix.png](http://url/of/the/path)
4. Summary: Summarize all the above evaluation results in tabular format.
5. Conclusion: Summarize this report, around 200 words.
Here is an example for you: 
{example}

'''

KNOWLEDGE_INTEGRATION_SYSTEM = '''\nAdditionally, you can retrieve the code from the knowledge base. You should refer to the knowledge code to complete the task if the code is provided. The retrieved code will be formatted as:
Retrieval: The retriever found the following pieces of code cloud address the problem. All functions and classes have been defined and executed in the back-end.
Core code (All functions and classes have been defined, directly use them in your code):
```core_function
{core}
```

Here is an example for you to do retrieval and knowledge integration:
User: I want to calculate the nearest correlation matrix by the Quadratically Convergent Newton Method. Please write a well-detailed code. The code gives details of the computation for each iteration, such as the norm of gradient, relative duality gap, dual objective function value, primal objective function value, and the running time.
Using the following parameters to run a test case and show the result:
Set a 2000x2000 random matrix whose elements are randomly drawn from a standard normal distribution, the matrix should be symmetric positive, and semi-definite.
Set the b vector by 2000x1 with all elements 1.
Set tau by 0.1, and tolerance error by 1.0e-7.
Retrieval: The retriever found the following pieces of code cloud address the problem.  All functions are well-defined and have been executed in the back-end.
Core code (All functions and classes have been defined, directly use them in your code):
```core_function
# test
n = 3000
data_g_test = scipy.randn(n, n)
data_g_test = (data_g_test + data_g_test.transpose()) / 2.0
data_g_test = data_g_test - np.diag(np.diag(data_g_test)) + np.eye(n)
b = np.ones((n, 1))
tau = 0
tol = 1.0e-6
[x_test_result, y_test_result] = NearestCorrelationMatrix(data_g_test, b, tau, tol)
print(x_test_result)
print(y_test_result)
```

Your code:
```
import numpy as np
from scipy import randn
# Define the input matrix
n = 3000
data_g_test = np.random.randn(n, n)
data_g_test = (data_g_test + data_g_test.transpose()) / 2.0
data_g_test = data_g_test - np.diag(np.diag(data_g_test)) + np.eye(n)
# Define the initial guess
b = np.ones((n, 1))
# Define the penalty parameter and tolerance
tau = 0
tol = 1.0e-6
# Call the NearestCorrelationMatrix function (Directly use NearestCorrelationMatrix())
[x_test_result, y_test_result] = NearestCorrelationMatrix(data_g_test, b, tau, tol) 
print(x_test_result)
print(y_test_result)
```
'''

# KNOWLEDGE_INJECTION_PMT_FULL = """
# \nThe retriever find the following pieces of code cloud address the problem. The base function is the sub-function will be used in the entrance function. The entrance funtcion is function that cater for user's request which can be used directly. Test case gives you a use case that you should refer to write the code.\n
# Base function: \n{base_function}\n
# Entrance code: \n{entrance_function}\n
# Test case: {test_case}\n
# """

PMT_KNW_IN_FULL = """
\n📝 Retrieval:\nThe retriever found the following pieces of code cloud address the problem. You may refer to this code and modify it as appropriate.
Retrieval code:\n```core_function{code}\n```\n
Your modified code:
"""

# PMT_KNW_IN_CORE = """
# \nRetrieval: The retriever find the following pieces of code cloud address the problem. All runnable functions have been executed in the back-end. You should directly refer to the core function code and modify it as appropriate.\n
# Core function code: \n{core}\n\n
# Runnable function (already executed in back-end): \n```python\n{runnable}\n```
# Your modified code (wrapped in ```python ```):
# """

PMT_KNW_IN_CORE = """
\n📝 Retrieval:\nThe retriever found the following pieces of code cloud address the problem. All functions and classes have been defined and executed in the back-end.
Core code (All functions and classes have been defined, directly use them in your code):\n```core_function\n{core}\n```\n
Your code:
"""

# PMT_KNW_IN_CORE = """
# \n📝 Retrieval:\nThe retriever found the following pieces of code cloud address the problem. All functions are well-defined and have been executed in the back-end. So, you should directly refer to the core code and modify it as appropriate instead of re-implement any function in runnable function.\n
# Core code (All functions have been well-defined in the runnable function):\n```core_function\n{core}\n```\n
# Runnable function (Have been executed in back-end): \n```runnable\n{runnable}\n```
# Your modified code:
# """

# PMT_KNW_IN_CORE = """
# \nRetrieval: The retriever find the following pieces of code cloud address the problem. The core code is a well-written function code that can be used directly. Test case gives you a use case that you should refer to write the code. Assuming all methods are implemented.\n
# description: {describe}\n
# Entrance code: \n{entrance_function}\n
# Test case: \n{test_case}\n
# """

# KNOWLEDGE_INJECTION_PMT_FIXED = """
# \nThe retriever find the following pieces of code cloud address the problem. Test case gives you a use case that you should refer to write the code assuming all methods are implemented.\n
# Test case: \n{test_case}\n
# """

# KNOWLEDGE_INJECTION_PMT = """Repeat the following code:
# Entrance code: \n{entrance_code}\n
# Test case: \n{test_case}\n
# """

prompts_fc = {
    'prompt_data_description': 'For the given information, pleas show the statistical describe to user first. Then use the perspective of a data analyst to explain'
                               'to the user. Your answer should be comprehensive, clear, and easy to understand, so that'
                               'an inexperienced person can easily understand it. You can also provide some speculations, '
                               'but not limited to what task may this data be used for, such as classification, regression, clustering,'
                               'and other task, what the target of the data may be, how to process this data if modeling'
                               'is needed later, and so on.\n',
    'prompt_data_processing': 'For the given information of data processing, pleas use the perspective of a data analyst to describe and explain'
                              'to the user. Your answer should be comprehensive and clear. Based on your speculation, you can prompt '
                              'the user on what to do next. Please prompt the user to do datasets partitioning in the last step of data processing,'
                              'so that could simplify the processing process.\n',
    'prompt_modeling': 'For the give model and information, pleas use the perspective of a data analyst to describe and explain'
                       'to the user. You can recommend hyper-parameters to user, or notice the user that they can do evaluation next.\n',
    'prompt_evaluation': 'For the given evaluation information, pleas use the perspective of a data analyst to describe and explain'
                         'to the user. If the result is not good, please give some suggestions that may improve the performance.\n',
    'prompt_document': 'For all the above historical information, pleas use the perspective of a data analyst to write an report. '
                       'The report should following the outline bellow:'
                       '1. title 2. backgournd, value 3. data description 4. data visiualization and analysis 5. data processing 6. modeling  '
                       '7. evaluation 8. conclusion\n'
}

if __name__ == '__main__':
    print(SYSTEM_PROMPT.format(local_path='aaaa'))
