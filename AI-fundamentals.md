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
- It assumes a **linear relationship** between:
  - Target variable (output)
  - Predictor variables (inputs/features)
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

> Basically:
> - Regression predicts **continuous numerical values**.
> - Linear Regression is a regression technique that assumes a **straight-line relationship** between inputs and output.
> - The objective is to find the line that best fits the data while minimizing prediction errors.
