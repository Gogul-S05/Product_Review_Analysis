# Product Reviews Sentiment Analysis using Transformers

## 📌 Project Overview

This project focuses on analyzing customer product reviews and classifying them into three sentiment categories:

* **Positive**
* **Neutral**
* **Negative**

The project uses a pre-trained **Sentence Transformer** model to convert customer reviews into sentence embeddings. These embeddings are then used as input features for machine learning classification models such as **Random Forest** and **Gradient Boosting**.

The objective is to help businesses understand customer feedback at scale and identify areas related to customer satisfaction, product issues, and improvement opportunities.

---

## 🎯 Business Objective

Customer reviews contain valuable information about product quality, usability, customer satisfaction, and perceived value. However, manually analyzing a large number of reviews is time-consuming.

This project provides an automated sentiment analysis approach that can help businesses:

* Identify positive customer feedback.
* Detect negative reviews and potential product issues.
* Analyze neutral feedback and possible unmet expectations.
* Support product development and customer service.
* Improve data-driven decision-making.

---

## 📊 Dataset

The dataset contains **Product Reviews** with the following columns:

| Column           | Description                                    |
| ---------------- | ---------------------------------------------- |
| `Product ID`     | Unique identification number for each product  |
| `Product Review` | Customer's review or opinion about the product |
| `Sentiment`      | Sentiment associated with the review           |

### Dataset Summary

* **Rows:** 1007
* **Columns:** 3
* **Missing values:** None
* **Duplicate rows:** 2 duplicates were identified and removed.

The sentiment distribution contains a majority of **positive reviews**, while neutral and negative reviews are considerably fewer.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Sentence Transformers
* Google Colab

### Machine Learning Models

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**

### Transformer Model

**Sentence Transformer:** `all-MiniLM-L6-v2`

---

## 🔄 Project Workflow

```text
Product Reviews Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Exploratory Data Analysis
        ↓
Sentence Transformer
        ↓
Sentence Embeddings
        ↓
Train-Test Split
        ↓
Random Forest & Gradient Boosting
        ↓
Model Evaluation
        ↓
Final Model Selection
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the product review dataset using Pandas.
2. Checked the dataset dimensions.
3. Examined the first few records.
4. Checked for missing values.
5. Identified duplicate records.
6. Removed duplicate rows.
7. Reset the DataFrame index.
8. Visualized the sentiment distribution.

---

## 🤖 Sentence Embeddings

The project uses the pre-trained Sentence Transformer model:

```python
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

The `Product Review` text is converted into numerical sentence embeddings.

These embeddings capture semantic information from the reviews and are used as input features for the machine learning models.

```python
embedding_matrix = model.encode(
    data['Product Review'].tolist(),
    show_progress_bar=True
)
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80:20 ratio**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` is used to maintain the sentiment class distribution between the training and testing datasets.

---

## 🌲 Random Forest

A Random Forest Classifier is trained using the Transformer-generated embeddings.

```python
rf_transformer = RandomForestClassifier(random_state=42)

rf_transformer.fit(X_train, y_train)
```

The model is evaluated using:

* Accuracy
* Weighted F1 Score

The notebook reports approximately **86% test accuracy** and **82% weighted F1 score** for the Random Forest + Transformer approach.

---

## 📈 Gradient Boosting

A Gradient Boosting Classifier is also trained using the generated sentence embeddings.

```python
gb_transformer = GradientBoostingClassifier(random_state=42)

gb_transformer.fit(X_train, y_train)
```

The notebook reports approximately **84% test accuracy** and **80% weighted F1 score** for the Gradient Boosting + Transformer approach.

---

## 📏 Evaluation Metrics

### Accuracy

Accuracy measures the proportion of correctly classified reviews among all predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### F1 Score

F1 Score provides a balance between precision and recall. A **weighted F1 score** is used to account for the different numbers of samples in each sentiment class.

---

## 📊 Model Performance

| Model                           | Test Accuracy | Test F1 Score |
| ------------------------------- | ------------: | ------------: |
| Random Forest + Transformer     |          ~86% |          ~82% |
| Gradient Boosting + Transformer |          ~84% |          ~80% |

The notebook selects **Random Forest + Transformer** as the final model based on the reported test performance and generalization comparison.

---

## 💡 Key Insights

* Customer reviews can be converted into meaningful numerical representations using Transformer-based sentence embeddings.
* The dataset is highly dominated by positive reviews.
* Random Forest and Gradient Boosting can effectively use Transformer embeddings for sentiment classification.
* The Random Forest model achieved the stronger reported test performance among the models evaluated.
* The difference between training and testing performance indicates that generalization should be considered when evaluating the models.

---

## 🚀 Future Improvements

The project can be further improved by:

* Adding more product review data.
* Handling class imbalance using appropriate techniques.
* Testing additional models such as XGBoost and Support Vector Machines.
* Fine-tuning a Transformer model specifically for sentiment classification.
* Continuously retraining the model using newly collected customer reviews.
* Monitoring changes in customer language and sentiment over time.
* Performing more detailed error analysis on incorrectly classified reviews.

---

## 📁 Project Structure

```text
Product-Review-Sentiment-Analysis/
│
├── Product_Review_Sentiment_Analysis_Transformers.ipynb
├── Product_Reviews.csv
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
Product_Review_Sentiment_Analysis_Transformers.ipynb
```

using **Google Colab** or **Jupyter Notebook**.

### 3. Install the required package

```python
!pip install sentence-transformers
```

### 4. Restart the runtime

If using Google Colab, restart the runtime after installing the package and then execute the notebook cells sequentially.

### 5. Provide the dataset

Place `Product_Reviews.csv` in the required dataset location or update the CSV path in the notebook.

---

## 👨‍💻 Project Skills Demonstrated

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Natural Language Processing
* Sentence Embeddings
* Transformer Models
* Machine Learning
* Random Forest
* Gradient Boosting
* Model Evaluation
* Sentiment Analysis
* Data Visualization

---

## 🏁 Conclusion

This project demonstrates how **Transformer-based sentence embeddings** can be combined with traditional machine learning algorithms to classify product reviews into positive, neutral, and negative sentiments.

The evaluated models demonstrate that semantic representations generated by Sentence Transformers can be effectively used for downstream sentiment classification. The project can be extended with additional data, advanced models, and Transformer fine-tuning to further improve sentiment analysis capabilities.
