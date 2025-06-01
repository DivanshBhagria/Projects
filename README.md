# Projects
# Machine Learning Algorithms Implementation

This repository contains Jupyter Notebook implementations of essential machine learning algorithms and data preprocessing techniques. Below is a detailed overview of each notebook, organized in a logical learning progression.

## Table of Contents

### 1. [Apriori Algorithm](1.Apriori_(Associate_rule_learning).ipynb)
**File:** `1.Apriori_(Associate_rule_learning).ipynb`  
**Usage:** Market basket analysis using association rule mining  
**Features:**
- Implements Apriori algorithm for product association discovery
- Identifies relationships between frequently purchased items
- Visualizes rules based on support, confidence, and lift metrics
- Applications: Recommendation systems, store layout optimization
- Output: Product pairs with highest association strength

### 2. [Artificial Neural Network](2.Artificial_neural_network.ipynb)
**File:** `2.Artificial_neural_network.ipynb`  
**Usage:** Customer churn prediction using deep learning  
**Features:**
- Implements ANN with TensorFlow/Keras
- Data preprocessing with one-hot encoding
- Binary classification for customer retention
- Hyperparameter tuning with epochs and batch size
- Model evaluation using confusion matrix
- Single customer prediction example

### 3. [Convolutional Neural Network](3.Convolution_Neural_Network_(CNN).ipynb)
**File:** `3.Convolution_Neural_Network_(CNN).ipynb`  
**Usage:** Image classification (cats vs dogs)  
**Features:**
- Implements CNN architecture with convolutional/pooling layers
- Image augmentation techniques (shearing, zooming, flipping)
- Handles image preprocessing and transformation
- Binary classification with sigmoid activation
- Single image prediction functionality
- Visualizes training/validation accuracy

### 4. [Eclat Algorithm](4.Eclat_(Association_learning).ipynb)
**File:** `4.Eclat_(Association_learning).ipynb`  
**Usage:** Simplified association rule learning  
**Features:**
- Implements Eclat algorithm for frequent itemset mining
- Identifies product pairs with high support values
- Optimized version of association rule learning
- Applications: Market basket analysis, cross-selling
- Output: Product pairs sorted by support value
- Requires same dataset as Apriori

### 5. [Sentiment Analysis](5.Sentiment_Analysis.ipynb)
**File:** `5.Sentiment_Analysis.ipynb`  
**Usage:** Restaurant review classification using NLP  
**Features:**
- Text preprocessing pipeline (cleaning, stemming)
- Custom stopword handling with exception for 'not'
- Bag-of-Words model implementation
- Naive Bayes classifier for sentiment classification
- Confusion matrix for performance evaluation
- Handles real-world text data challenges

### 6. [XGBoost Classifier](6.XG_Boost.ipynb)
**File:** `6.XG_Boost.ipynb`  
**Usage:** Classification using gradient boosting  
**Features:**
- Implements XGBoost algorithm
- Includes k-Fold cross-validation
- Accuracy metrics with standard deviation
- Confusion matrix evaluation
- Handles both numerical and categorical features
- Efficient implementation for structured data

### 7. [Machine Learning Data Preprocessing](7.ML_Data_PreProcessing.ipynb)
**File:** `7.ML_Data_PreProcessing.ipynb`  
**Usage:** Essential data preparation pipeline  
**Features:**
- Comprehensive data cleaning and transformation
- Missing value handling with mean imputation
- Categorical encoding (One-Hot and Label Encoding)
- Feature scaling using StandardScaler
- Train-test splitting with randomization
- Foundation for all machine learning workflows

---

## Installation and Usage

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/machine-learning-implementations.git
cd machine-learning-implementations
```

2. **Install required dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk apyori xgboost jupyter
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

4. **Run notebooks in numerical order** for progressive learning experience

---

## Dataset Requirements

| Notebook | Required Dataset | Download Link |
|----------|------------------|---------------|
| 1. Apriori | `Market_Basket_Optimisation.csv` | [Download](https://www.kaggle.com/datasets/heeraldedhia/market-basket-optimisation) |
| 2. ANN | `Churn_Modelling.csv` | [Download](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| 3. CNN | Cats vs Dogs images | [Download](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) |
| 4. Eclat | `Market_Basket_Optimisation.csv` | Same as Apriori |
| 5. Sentiment Analysis | `Restaurant_Reviews.tsv` | [Download](https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews) |
| 6. XGBoost | `Data.csv` | Included in repository |
| 7. Data Preprocessing | `Data.csv` | Included in repository |

**Note:** Place all datasets in the same directory as the notebooks before execution.

---

## Key Features Across All Notebooks
- **Hands-on Implementation**: Practical code examples for each algorithm
- **Real-world Applications**: Solutions to business problems like customer churn, sentiment analysis
- **Detailed Comments**: Step-by-step explanations of each code block
- **Visualizations**: Includes plots and charts for result interpretation
- **Self-contained**: Each notebook runs independently with minimal setup
- **Progressive Structure**: Ordered from fundamental to advanced techniques

![Machine Learning Workflow](https://miro.medium.com/max/1400/1*VVD1q6cVKN3g8g8ZbE5x5A.png)

## Contribution
Contributions are welcome! Please open an issue or submit a pull request for:
- Additional algorithms
- Dataset suggestions
- Performance improvements
- Documentation enhancements
