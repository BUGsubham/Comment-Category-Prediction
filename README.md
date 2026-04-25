# Comment Category Prediction

Multi-class classification project for predicting a comment category label from text, metadata, and social interaction signals.

## Overview

This repository contains a notebook-based machine learning pipeline that:

1. Loads train and test datasets
2. Performs exploratory data analysis (EDA)
3. Engineers datetime and text-derived features
4. Builds a mixed-feature preprocessing pipeline
5. Trains and evaluates multiple classifiers
6. Produces predictions in competition submission format

The current workflow is implemented in [github_notebook.ipynb](github_notebook.ipynb).

## Problem Type

- Task: Multi-class classification
- Target: label (classes 0, 1, 2, 3)
- Input modalities:
	- Free-text comment
	- Numeric interaction features (for example upvotes/downvotes, if_1, if_2)
	- Categorical demographic/context fields (for example race, religion, gender)
	- Binary field(s) (for example disability)
	- Timestamp (created_date)
- Challenge is to deal with data imbalance.
- Out of four, two classes comprise of 90% of the data.

## Repository Structure

```text
Comment-Category-Prediction/
|-- github_notebook.ipynb
|-- README.md
`-- data/
		|-- train.csv
		|-- test.csv
		|-- Sample.csv
		`-- submission.csv
```

## Data Schema

### Train Columns

- created_date
- post_id
- emoticon_1
- emoticon_2
- emoticon_3
- upvote
- downvote
- if_1
- if_2
- race
- religion
- gender
- disability
- comment
- label


## Methodology Used In Notebook

### 1) EDA

- Label distribution analysis (class imbalance is present)
- Numeric distribution summary and boxplots
- Categorical frequency inspection

### 2) Feature Engineering

- Datetime features from created_date:
	- year, month, day, dayofweek, quarter, is_weekend
- Text cleaning on comment:
	- lowercase normalization
	- URL/email/number token replacement
- Additional text statistics:
	- comment_len, comment_word_count, comment_avg_word_len
	- uppercase ratio, punctuation and symbol-based counts/ratios

### 3) Preprocessing

- Numeric features: StandardScaler
- Categorical features: most-frequent imputation + OneHotEncoder
- Text features:
	- word TF-IDF (1-2 grams)
	- character TF-IDF (3-5 char n-grams)
	- dimensionality reduction with TruncatedSVD
- Combined with ColumnTransformer

### 4) Models

- LogisticRegression with class weighting
- LightGBM (LGBMClassifier) with tuned boosting parameters
- Soft Voting ensemble over LogisticRegression + LightGBM

### 5) Evaluation

- Stratified train/validation split
- Classification report
- Normalized confusion matrix
- Error-focused normalized confusion matrix

