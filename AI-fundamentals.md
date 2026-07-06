# Intro to ML:
<img width="1500" height="1500" alt="image" src="https://github.com/user-attachments/assets/188366c9-8163-4cc9-8cd7-691a650b8e7b" />
### AI:
- Ai Is broad field focusing on developing systems that can perform tasks requiring human intelligence.
- Key areas of AI:
  - **Natural Language Processing (NLP):** Enabling computers to understand, interpret, and generate human language.
  - **Computer Vision:** Allowing computers to "see" and interpret images and videos.
  - **Robotics:** Developing robots that can perform tasks autonomously or with human guidance.
  - **Expert Systems:** Creating systems that mimic the decision-making abilities of human experts.

### ML:
- Subfield of AI that learns from data and improve their performance on specific tasks without writing code.
- ML algorithms categorized into 3 types:
  - Supervised Learning: Learns from labeled data where each data point is known outcome. (e.g. image classification, spam detection, fraud prevention)
  - Unsupervised Learning: Learns from unlabled data without providing outcome or label. (e.g. customer segmentation, anomaly detection, Dimensionality reduction - i don't know meaning of this)
  - Reinforcement Learning:  Runs through trial and error by interacting with an env and receiving feedback as reward or penalty. (e.g. game playing, robotics, autonomous driving)

### Deep Learning (DL):
- Subfield of ML that uses neural networks with multiple layers to learn and extract features from complex data.
- Deep neural networks automatically identify intricate patterns and representations within large datasets. Which makes it powerful for taks with unstructured or HD data (images, audio, text etc...)
- Key characteristics of ML:
  - Hierarchical Feature Learning: DL models can learn hierarchical data representations, where each layer captures abstract features. e.g. lower layer detects edges and textures in image while upper layer detects complex features as shapes and objects.
  - End-to-End Learning: DL models can be trained end-to-end, meaning they can directly map raw input data to desired outputs without manual feature engineering.
  - Scalability: DL models can scale well with large datasets and computational resources, making them suitable for big data applications.
- Common types of neural networks used in DL:
  - **Convolutional Neural Networks (CNNs):** Specialized for image and video data
  - **Recurrent Neural Networks (RNNs):** Designed for sequential data like text and speech
  - **Transformers:** A recent advancement in DL, effective for natural language processing tasks.

### Relationship between AI,ML and DL:
- ML and DL are subfield of AI that enables systems to learn from data and make intelligent decisions.
- DL has enhanced the capabilities of ML by providing powerful tools for feature extraction and representation learning, especially in domains with complex and unstructured data.

## Mathematics refresher of AI:
- Maths concepts behind algorithms and processes. It's just a reference.
- Basic Arithmetic Operations: Multiplication (`*`), Division (`/`), Addition (`+`), Subtraction (`-`)
- Algebraic Notations: Subscript Notation (`x_t`), Superscript Notation (`x^n`), Norm (`||...||`), Summation Symbol (`Σ`)
- Logarithms and Exponentials: Logarithm Base 2 (`log2(x)`), Natural Logarithm (`ln(x)`), Exponential Function (`e^x`), Exponential Function (Base 2) (`2^x`)
- Matrix and Vector Operations:
  - Matrix-Vector Multiplication (`A * v`) => `A * v = [ [1, 2], [3, 4] ] * [5, 6] = [17, 39]`,
  - Matrix-Matrix Multiplication (`A * B`) => `A * B = [ [1, 2], [3, 4] ] * [ [5, 6], [7, 8] ] = [ [19, 22], [43, 50] ]`,
  - Transpose (`A^T`) => swaps the rows and columns of A. `A = [ [1, 2], [3, 4] ] \n A^T = [ [1, 3], [2, 4] ]`,
  - Inverse (`A^{-1}`) => matrix that, when multiplied by A, results in the identity matrix. `A = [ [1, 2], [3, 4] ]  \n  A^{-1} = [ [-2, 1], [1.5, -0.5] ]`,
  - Determinant (`det(A)`) => `A = [ [1, 2], [3, 4] ] \n det(A) = 1 * 4 - 2 * 3 = -2`,
  - Trace (`tr(A)`) => sum of the elements on the main diagonal. `A = [ [1, 2], [3, 4] ] \n tr(A) = 1 + 4 = 5`
- Set Theory:
  - Cardinality (`|S|`) => represents the number of elements in a set S. `S = {1, 2, 3, 4, 5} then |S| = 5`,
  - Union (`∪`) => The union of two sets A and B. `A = {1, 2, 3}, B = {3, 4, 5} then A ∪ B = {1, 2, 3, 4, 5}`,
  - Intersection (`∩`) => The intersection of two sets. `A = {1, 2, 3}, B = {3, 4, 5} then A ∩ B = {3}`,
  - Complement (`A^c`) => set of all elements not in A. `U = {1, 2, 3, 4, 5}, A = {1, 2, 3} then A^c = {4, 5}`
- Comparison Operators: Greater Than or Equal To (`>=`), Less Than or Equal To (`<=`), Equality (`==`), Inequality (`!=`)
- Eigenvalues and Scalars:
  - Lambda (Eigenvalue) (`λ`) => λ represents an eigenvalue in linear algebra or a scalar parameter in equations. `A * v = λ * v, where λ = 3`,
  - Eigenvector => non-zero vector that, when multiplied by a matrix, results in a scalar multiple of itself. The scalar is the eigenvalue. `A * v = λ * v`
- Functions and Operators: Maximum Function (`max(...)`), Minimum Function (`min(...)`), Reciprocal (`1 / ...`), Ellipsis (`...`) is infinite process (`a_1 + a_2 + ... + a_n`)
- Functions and Probability: Function Notation (`f(x)`), Conditional Probability Distribution (`P(x | y)`), Expectation Operator (`E[...]`), Variance (`Var(X)`), Standard Deviation (`σ(X)`), Covariance (`Cov(X, Y)`), Correlation (`ρ(X, Y)`)

> Basically AI is a whole maths at core level.

# Supervised Learning Algos:
- Learn from labeled data. Each data point is associated with known outcome.
- This algorithm aims to learn mapping function to predict the label for new data. (Identify patterns and relationship between inputs and outputs.)

### How supervised learning works:
- Example: a child identifying different fruits. By repeatedly presenting examples with labels, kid learns to identify based on color, shape, size etc...
- Algo works similarly. Its fed with large datasets of labeled examples and they use this to train model.

- Supervised learning problems:
  1. Classification: In classification problems, the goal is to predict a categorical label. For example, classifying emails as spam or not or identifying images of cats, dogs, or birds.
  2. Regression: In regression problems, the goal is to predict a continuous value. For example, one could predict the price of a house based on its size, location, and other features or forecast the stock market.

### Core Concepts:
- Training Data: Foundation pf supervised learning. Labeled dataset used to train the ML model. Quality and quantity of training data impacts model's accuracy and ability to generalize new data.
- Features: Measurable properties of data that serves as input to models. They are variables that algorithm uses to learn and make predictions. Selecting relevant features is imp for building effective model.
- Labels: Known outcomes associated with each data point. They represent the "correct answers" that the model aims to predict.
- Model: Mathematical representation of the relationship between the features and the labels. It's learned from training data and used to predict new data. Model can be considered function that takes input and predicts output.
- Training: Process of feeding training data to the algorithm and adjusting model's parameters to minimize prediction errors. The algo learns from training data by iteratively adjusting parameters to get better predictions.
- Prediction: Once the model is trained, it's used to predict new data. It involves providing model with features and model will predict output. Prediction is specific app of inference, focusing on generating specific output such as classifying an email as spam etc.
- Inference: Broader concept that encompasses prediction but also includes understading structures and patterns in data. It involves using a trained model to derive insights, estimate parameters and understand relationship between variables. e.g. Determining which features are more imp in a decision tree.
- Evaluation: Critical step is supervised learning. It involves accessing models performance to determine accuracy and generation ability to new data. Common evaluation matrics:
  - **Accuracy:** Proportin of correct predictions made by the model.
  - **Precision:** Proportion of true positive predictions out of all positive predictions.
  - **Recall:** Proportion of true positive predictions out of all actual positive instances.
  - **F1-Score:** Harminic mean of precision and recall, providing balanced measure of model's performance.
- Generalization: Refers to the model's ability to accurately predict outcomes for new data not used during training. A model that generalizes well can apply its learned knowledge to real-world scenarios.
- Overfitting: When model learns training data too well, including noise and outliners. It can lead to poor generalization of new data as model memorized training sets instead of learning underlying patterns. 
- Underfitting: When model is too simple to capture underlying patterns in data. It leads to poor performance in both training and new data.
- Cross-Validation: Technique used to assess how well a model will generalize to independent dataset. It involves spilliting data into mutiple subsets (folds) and then train model on different combinations of these folds while validating it on remaining folds. It helps reduce overfitting and provide more reliable outcomes.
- Regularization: Technique used to prevent overfitting by adding a penalty term to the loss function. This penalty discourages model from learning too complex patterns that might not generalize well. Common regularization techniques:
  - L1 : Adds a penalty equal to the absolute value of the magnitude of coefficients.
  - L2 : Adds a penalty equal to the square of the magnitude of coefficients.

## Linear Regression:
<img width="685" height="548" alt="Liner Regression Model" src="https://github.com/user-attachments/assets/db433d60-4713-4216-8c9a-5dbc5dea8e8e" />

- It is used to predict a **continuous target variable**.
- It assumes a **linear relationship** between `Predictor variables (inputs/features)` and `Target variable (output)`.
- The relationship is represented using a **linear equation**.
- Changes in predictor variables cause proportional changes in the target variable.

#### Goals of Linear Regression:
- Find the **best-fitting straight line** that represents the relationship between input and output variables.
- Minimize the **sum of squared differences** between:
  - Actual values
  - Predicted values
- This optimization method is known as **Least Squares**.

#### Example: House price prediction
###### Problem: 
- Predict a house's price based on its size.
###### How linear regression works?
- Linear Regression tries to find a straight line that best describes the relationship between:
  - House size (predictor variable)
  - House price (target variable)
- Generally:
  - As house size increases, price tends to increase.
- Once the relationship is learned, the model can predict the price of a house given its size.

### What is Regression?
- Type of supervised learning. It's goal is to predict a **continuous target variable**. The output can take any vaslue within the range.
- Regression focuses on estimating numerical values rather than assigning categories.
- The key idea is:
  - **Regression → Predicts numbers**
  - **Classification → Predicts categories/labels**

#### Examples of regression problems:
1. House Price Prediction
- Predict house prices using:
  - Size
  - Location
  - Age
2. Temperature Forecasting
- Predict daily temperature using:
  - Historical weather data
3. Website Traffic Estimation
- Predict number of website visitors using:
  - Marketing spend
  - Time of year

#### Regression vs Classification:

| Regression | Classification |
|------------|---------------|
| Predicts continuous values | Predicts categorical labels |
| Output is a number | Output is a class/category |
| Example: House price prediction | Example: Spam vs Not Spam |
| Example: Temperature prediction | Example: Fraud vs Not Fraud |

##### Linear regression as type of regression:
- Linear regression is specific type of regression analysis. It assumes linear relationship between `predictor variables` and `target variable`.
- The relationship is modeled using a **straight line**.
- Used when the relationship between variables can be approximated linearly.

### Simple Linear Regression:
- Uses `one predictor variable (X)` and `one traget variable (Y)`.
- **Equation:**
```py
y = mx + c
```
- **Components:**
  - **y** → Predicted target variable
  - **x** → Predictor variable
  - **m** → Slope of the line
    - Represents change in y for a unit change in x
  - **c** → Y-intercept
    - Value of y when x = 0
- **Objective:**
  - Find optimal values of **m** `(slope)` and **c** `(intercept)`
  - Minimize prediction error between actual and predicted values.
- **Training Method:** Typically solved using **Ordinary Least Squares (OLS)**.

### Multiple Liner Regression: 
- Used when multiple predictor variables affect a single target variable.
- **Equation:**
```py
y = b0 + b1x1 + b2x2 + ... + bnxn
```
- **Components:**
  - **y** → Predicted target variable
  - **x1, x2, ..., xn** → Predictor variables
  - **b0** → Y-intercept
  - **b1, b2, ..., bn** → Coefficients of predictor variables
- **Interpretation:**
  - Each coefficient measures the influence of its predictor variable on the target variable.
  - Positive coefficient:
    - Increase in predictor → Increase in target
  - Negative coefficient:
    - Increase in predictor → Decrease in target

### Ordinary Least Squares (OLS):
<img width="840" height="702" alt="liner regression with residuals" src="https://github.com/user-attachments/assets/6c7d9d4c-fbf3-4d36-9aed-f4eb43e51c69" />

- Most common method for estimating coefficients in Linear Regression.
- Finds the line that best fits the data.
- **Objective:**
  - Minimize the difference between actual and predicted values.
  - Specifically minimizes the **Residual Sum of Squares (RSS)**.
- **Key terms:**
  - Residual: Difference between actual and predicted value. `Residual = Actual Value - Predicted Value`.
  - Residual Sum of Squares (RSS): Sum of all squared residuals. `RSS = Σ(Actual - Predicted)²`.
    - Measures total prediction error.
    - Lower RSS indicates a better-fitting model.
- **Intuition:**
  - Imagine drawing squares between each data point and the regression line.
  - OLS finds the line that minimizes the total area of these squares.

 #### OLS Process:
1. Calculate Residuals:
```text
Residual = Actual - Predicted
```
2. Square Residuals:
  - Makes all errors positive.
  - Gives greater weight to larger errors.
3. Sum Squared Residuals:
```text
RSS = Σ Residual²
```
4. Minimize RSS:
  - Adjust model coefficients.
  - Find the smallest possible RSS.
  - Resulting line is called the **Line of Best Fit**.

### Assumptions of Linear Regression:
- Linear Regression relies on several assumptions for valid and reliable predictions.
1. Linearity:
   - Assumption: A linear relationship exists between predictor variables and target variable.
   - Meaning: Changes in inputs produce proportional changes in output.
2. Independence: 
   - Assumption: Observations are independent of each other.
   - Meaning: One observation should not influence another.
3. Homoscedasticity:
   - Assumption: Error variance remains constant across all levels of predictor variables.
   - Meaning: Residuals should have roughly equal spread throughout the dataset.
   - Desired pattern: `Residual Spread ≈ Constant` 
4. Normality:
   - Assumption: Residuals (errors) follow a normal distribution.
   - Importance: Required for Statistical inference, Confidence intervals, Hypothesis testing, Reliable coefficient interpretation

##### Why assumption matter: 
If assumptions are violated:
- Predictions may become inaccurate.
- Coefficients may be misleading.
- Statistical conclusions may be unreliable.

> **Summary:**
> - Linear Regression predicts continuous numerical values.
> - It assumes a linear relationship between input and output variables.
> - Simple Linear Regression uses one predictor variable.
> - Multiple Linear Regression uses multiple predictor variables.
> - OLS finds the best-fit line by minimizing RSS.
> - Key assumptions:
>   1. Linearity
>   2. Independence
>   3. Homoscedasticity
>   4. Normality

## Logistic Regression:
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/749991be-bc4e-43df-b50d-e2ad2a09a022" />

- Mainly used for **classification**, not regression.
- Predicts variable with two possible outcomes (binary classification).
- Binary representations `0 or 1`, `true or false`, `yes or no`.
- Example: predict whether email is spam or not.
- The algo models probability of target variable belongs to a particular class using logistic function.
- Logistic Function: Maps the input features to a value between 0 and 1.

### What is Classification:
- Type of supervised learning.
- Assign data points to specific category or classes.
- Predict **labels**, not values (predicted in regression).

#### Examples of Classification problems:
- Identifying fraudulent transactions (fraudulent or not fraudulent)
- Classifying images of animals (cat, dog, bird, etc.)
- Diagnosing diseases based on patient symptoms (disease present or not present)

### How Logistic Regression works??
- Logistic regression outputs a probability score between 0 and 1.
- It achieves this by employing `sigmoid function`.
- **Sigmoid Function** introduces non-linearity, allowing the model to capture complex relationships between the features and the probability of the outcome.

### What is sigmoid function:
<img width="565" height="456" alt="image" src="https://github.com/user-attachments/assets/501b227b-d974-4ede-860c-ba1d58851776" />

- Mathemetical function that takes any input value (ranging from negative to positive infinity) and maps it to an output value between 0 and 1.
- Useful for modeling probabilities.
- It has a characteristic "S" shape.
- In logistic regression, this function transforms the linear combination of input features into a probability score. This score represents the likelihood of the input belonging to the positive class.

##### Sigmoid function mathematical representation:
```
P(x) = 1 / (1 + e^-z)
```
Where: 
- `P(x)` = predicted probability.
- `e` = base of the natural logarithm (approximately 2.718).
- `z` = linear combination of input features and their weights, similar to the linear regression equation: `z = m1x1 + m2x2 + ... + mnxn + c`

### Spam Detection:
- Example when building a spam filter using logistic regression, the algo will analyse the email features such as sender address, certain keywords, email content, and then calculate the probability score.
- And the email will be classified as spam if the score exceeds defined threshold (e.g. 0.8).

### Decision Boundary:
<img width="844" height="701" alt="image" src="https://github.com/user-attachments/assets/61c5ba0f-5ada-4d4e-a5db-24ba7f9b2385" />

- **Decision Boundary:** A line seperating data points into 2 classes.
- This seperator boundary is determined by model's learned parameters and the chosen threshold probability.
- In higher dimensions with more features, this separator becomes a hyperplane.

### Hyperplanes:
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/cd82c47c-055a-45a6-8b4a-00e05e95c0ff" />

- In the context of ML, hyperplane is a subspace whose dimension is one less than that of the ambient space.
- It's a way to visualize a decision boundary in higher dimensions.
- A hyperplane is a "flat" subspace that divides the higher-dimensional space into two regions.
- For example, in 2D (paper) => it's a line that divides the space in 2 parts. And in 3D (room) => it's a flat plane to divide space.
- Visualizing more than 3D is difficult, but the concept remains the same.

### Threshold Probability:
- Often set at `0.5` but adjusted depending upon specific problem and desired balance between true and flase positives.
- If data points predicted probability is `above threshold`, then `positive class`. If `below threshold`, then `negative class`.
- For example, in spam detection, if the model predicts an email has a 0.8 probability of being spam (and the threshold is 0.5), it's classified as spam.

### Data Assumptions:
