# Multi-Class Emotion-Sentiment Analysis using ModernBERT and Hugging Face 🤖

This repository contains a complete end-to-end project for fine-tuning a `ModernBERT` model for multi-class emotion classification on tweet data. The entire workflow is built using the Hugging Face ecosystem, including `transformers`, `datasets`, and the `Trainer` API for efficient training and evaluation.

The goal is to classify tweets into one of six emotions: **Normal (Joy), Sadness, Anger, Fear, Love,** and **Surprise**.

-----

## ✨ Key Features

  * **State-of-the-Art Model**: Utilizes `answerdotai/ModernBERT-base`, a powerful and modern variant of the classic BERT model.
  * **Handling Class Imbalance**: Demonstrates a practical approach to addressing imbalanced datasets using undersampling to prevent model bias.
  * **End-to-End Workflow**: Covers every step from data loading and cleaning to tokenization, training, evaluation, and inference.
  * **Hugging Face Integration**: Leverages the high-level `Trainer` API for streamlined training and the `pipeline` function for easy, production-ready inference.
  * **Detailed Evaluation**: Includes a comprehensive evaluation using a classification report and a confusion matrix to visualize model performance across different emotions.

-----

## ⚙️ Project Workflow

The project follows a structured machine learning pipeline:

1.  **Environment Setup**: Installs all necessary libraries from the Python ecosystem.
2.  **Data Loading & Inspection**: Loads the tweet dataset from a CSV file using `pandas` and inspects its structure and class distribution.
3.  **Data Balancing**: Identifies a significant class imbalance and corrects it by **undersampling** each emotion class to an equal number of samples (500 per class).
4.  **Exploratory Data Analysis (EDA)**: Visualizes the balanced class distribution and analyzes tweet length patterns across different emotions.
5.  **Tokenization**: Uses the `AutoTokenizer` corresponding to `ModernBERT` to convert raw text into a numerical format suitable for the model.
6.  **Data Formatting**: Splits the data into training (70%), validation (10%), and testing (20%) sets and converts them into the Hugging Face `DatasetDict` format.
7.  **Model Configuration & Training**:
      * Configures `AutoModelForSequenceClassification` with the correct number of labels.
      * Sets up `TrainingArguments` to define hyperparameters like learning rate, batch size, and the number of epochs.
      * Initializes the `Trainer` with the model, datasets, and a custom function to compute accuracy and F1-score.
      * Launches the training process.
8.  **Evaluation**:
      * Predicts on the unseen test set to get final performance metrics.
      * Generates a detailed `classification_report` and a `confusion_matrix` to assess the model's strengths and weaknesses.
9.  **Inference & Deployment**:
      * Saves the fine-tuned model artifacts.
      * Demonstrates how to perform inference on new, unseen text using a simple custom function and the high-level `pipeline` API.

-----

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  Install the required Python libraries:

    ```bash
    pip install -U pandas torch scikit-learn matplotlib seaborn
    pip install -U transformers accelerate datasets evaluate
    ```

-----

## ▶️ How to Use the Trained Model

The fine-tuned model is saved in the `Modern-bert-uncased-sentiment-model` directory. The easiest way to use it for prediction is with the Hugging Face `pipeline` function.

```python
from transformers import pipeline

# Load the saved model into a text-classification pipeline
classifier = pipeline('text-classification', model='Modern-bert-uncased-sentiment-model')

# Prepare some example texts
texts_to_classify = [
    "I am super happy today. I got it done. Finally!!",
    "I'm feeling a bit down and lonely.",
    "That is an absolutely terrifying prospect!",
    "I love spending time with my family."
]

# Get predictions
predictions = classifier(texts_to_classify)

# Print results
for text, pred in zip(texts_to_classify, predictions):
    print(f"Text: '{text}'")
    print(f"Predicted Emotion: {pred['label']}, Score: {pred['score']:.4f}\n")

```

### Expected Output

```
Text: 'I am super happy today. I got it done. Finally!!'
Predicted Emotion: Normal, Score: 0.9987

Text: 'I'm feeling a bit down and lonely.'
Predicted Emotion: sadness, Score: 0.9995

Text: 'That is an absolutely terrifying prospect!'
Predicted Emotion: fear, Score: 0.9982

Text: 'I love spending time with my family.'
Predicted Emotion: love, Score: 0.9986
```

-----

## 📊 Model Evaluation

The model was evaluated on a held-out test set. The performance is measured using **Accuracy** and the **weighted F1-Score**, which is suitable for multi-class classification problems. A detailed breakdown of precision, recall, and F1-score for each emotion is available in the classification report, and the confusion matrix below provides a visual representation of the model's predictive accuracy.

The diagonal of the matrix shows the number of correctly classified instances for each emotion, while off-diagonal values indicate misclassifications.

-----

## 📚 Dataset

This project uses the **Twitter Multi-Class Sentiment Dataset**. The data was sourced from the following public GitHub repository:

  * **Data Source**: [https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter\_multi\_class\_sentiment.csv](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_multi_class_sentiment.csv)

The dataset contains tweets labeled with various emotions. For this project, we focused on the six most frequent classes and balanced them to ensure a fair training process.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
