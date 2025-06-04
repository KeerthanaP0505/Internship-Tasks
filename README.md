🧠 Sentiment Analysis with Logistic Regression
This project performs basic sentiment analysis using TF-IDF vectorization and Logistic Regression on a small custom dataset of textual reviews.

📌 Project Overview
🔍 Goal: Classify text into positive, negative, or neutral sentiments.

🧰 Tech stack: Python, Scikit-learn, NLTK, pandas, seaborn, matplotlib

📊 Algorithm: Multiclass Logistic Regression using one-vs-rest strategy.

📁 Dataset
A small manually created dataset with 10 short text reviews labeled as:

Positive (1)

Negative (0)

Neutral (2)

Each review is labeled for supervised machine learning training.

🧼 Data Preprocessing
Custom clean_text() function includes:

Lowercasing

Removing URLs, mentions, hashtags

Removing punctuation and digits

Stopword removal using NLTK

⚙️ Steps
Text Cleaning

Label Encoding

Train-Test Split

TF-IDF Vectorization

Model Training (Logistic Regression)

Evaluation (Accuracy, Classification Report, Confusion Matrix)

📈 Output Example
Classification Report with precision, recall, and F1-score.

Confusion Matrix heatmap for visual performance check.

📦 Requirements
Install the required libraries using pip:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn nltk
Download stopwords:

python
Copy
Edit
import nltk
nltk.download('stopwords')
🧪 Sample Code
python
Copy
Edit
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
model = LogisticRegression(multi_class='ovr')
model.fit(X_train_tfidf, y_train)
