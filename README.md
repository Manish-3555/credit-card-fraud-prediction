# Credit Card Fraud Detection

This is my project where I tried to detect fraud in credit card transactions. The tricky
part is that the data is super imbalanced — out of 284,807 transactions, only 492 are
actually fraud. That's just 0.17%.

## Why this is hard

If I just built a model that says "not fraud" every single time, it would still be 99.8%
accurate — but it would catch zero fraud. So accuracy is basically useless here. Instead I
focused on precision, recall, F1-score, and ROC-AUC to actually see how well the model
finds fraud.

## About the dataset

I used the [Credit Card Fraud dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
Most of the columns (`V1` to `V28`) are already transformed with PCA for privacy reasons, so
I don't really know what they represent — only `Time` and `Amount` are normal, readable
columns.


## What I did

1. **Looked at the data first** — checked how imbalanced it really was, checked for
   duplicate rows, and looked at how `Amount` and `Time` were distributed.
2. **Dropped duplicate rows** before doing anything else.
3. **Split into train and test first**, before touching the imbalance problem at all. If
   you balance the data before splitting, some of that "fake" or duplicated data can leak
   into your test set and make your model look better than it actually is. So I split first,
   kept the test set untouched, and only fixed the imbalance on the training data.
4. **Scaled `Amount` and `Time`** using RobustScaler. I picked this over the usual
   StandardScaler because `Amount` has a lot of extreme values (some of the fraud
   transactions literally are the outliers), and RobustScaler doesn't get thrown off by
   that the way StandardScaler does.
5. **Fixed the imbalance using SMOTE** — this creates new, synthetic fraud examples instead
   of just copying the same 492 fraud rows over and over. Only applied to the training data.
6. **Trained 4 models**: Logistic Regression, Decision Tree, Random Forest, and XGBoost, and
   compared them on the untouched test set.

## Results

Tested on 56,746 transactions (95 of them actual fraud):

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.05 | 0.87 | 0.10 | 0.960 |
| Decision Tree | 0.39 | 0.69 | 0.50 | 0.847 |
| **Random Forest** | **0.91** | 0.78 | **0.84** | 0.959 |
| XGBoost | 0.75 | **0.80** | 0.78 | **0.965** |

## What I actually found

**Random Forest** worked best overall for me — it only messed up 7 legitimate transactions
(false alarms) while still catching 78% of the actual fraud. That's a really good tradeoff.

**XGBoost** technically had the best ROC-AUC and caught slightly more fraud, but it also had
more false alarms. I think this is because I didn't tune it at all, and boosting models
tend to be more sensitive to the "fake" synthetic data SMOTE creates, especially near the
border where fraud and non-fraud transactions look similar.

**Logistic Regression** caught the most fraud (87%) but flagged 1,463 transactions as fraud
that weren't. That's way too many false alarms to actually use in real life.

So basically — there's no single "best" model here. It depends on what matters more: not
missing fraud, or not annoying too many real customers with false alarms.

## What I'd do next if I had more time

- Actually tune XGBoost's settings instead of using defaults — I think it can beat Random
  Forest with some tuning
- Try `scale_pos_weight` or `class_weight` instead of SMOTE and see if that works better
- Try to run a neural network model also

## Tools I used

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, matplotlib, seaborn

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook
```
Download the dataset first (link above), put it in this folder, then just run all the cells.
