import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import logging

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Logger setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Data Preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Text transformation function
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Data preprocessing function
def preprocess_df(df, text_column="text", target_column="target"):
    try:
        logger.debug("STARTING PREPROCESSING")
        encoder = LabelEncoder()
        df["target"] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        df[text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

# Main function to load, process, and save data
def main(text_column="text", target_column="target"):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded")

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
