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
- Training: 
