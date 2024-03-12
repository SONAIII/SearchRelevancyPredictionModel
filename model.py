#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_recall_curve, f1_score, PrecisionRecallDisplay

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


# In[148]:


training_df = pd.read_csv('data/train_df.csv')
test_df = pd.read_csv('data/test_df.csv')


# In[149]:


training_df


# In[150]:


y = training_df['target']
X = training_df.drop(['target', 'search_id'], axis=1)


# In[151]:


X.isnull().sum()


# In[152]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[153]:





# In[154]:





# In[155]:




rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
rf_model.fit(X_train, y_train)


y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"ROC-AUC Score: {roc_auc_rf}")


accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf}")
print(classification_report(y_test, y_pred_rf))


precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
plt.plot(recall_rf, precision_rf, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve - Random Forest')
plt.legend()
plt.show()


# # In[156]:





# param_grid = {
#     'n_estimators': [100, 200, 300], # Number of trees in the forest
#     'max_depth': [10, 20, 30],       # Maximum depth of the trees
#     'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
#     'min_samples_leaf': [1, 2, 4],   # Minimum number of samples required at a leaf node

# }

# rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_macro', verbose=2)

# grid_search.fit(X_train, y_train)

# # Print the best parameters and the best score
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")

# # Use the best estimator to predict on the test set
# y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
# y_pred = grid_search.best_estimator_.predict(X_test)

# # Evaluate the best model from grid search
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"ROC-AUC Score: {roc_auc}")
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(classification_report(y_test, y_pred))




# from sklearn.metrics import precision_recall_curve

# # Get the probability predictions for the minority class
# y_scores = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

# # Calculate precision and recall for various thresholds
# precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
# best_f1_index = np.argmax(fscores)
# best_threshold = thresholds[best_f1_index]
# best_f1_score = fscores[best_f1_index]

# print(f"Best threshold: {best_threshold}")
# print(f"Best F1 Score: {best_f1_score}")

# y_pred_adjusted = (y_scores >= best_threshold).astype(int)


# print(f"Adjusted Accuracy: {accuracy_score(y_test, y_pred_adjusted)}")
# print(classification_report(y_test, y_pred_adjusted))


# In[158]:


y_final_test = test_df['target']
X_final_test = test_df.drop(['target', 'search_id'], axis=1)
y_scores_final_test = rf_model.predict_proba(X_final_test)[:, 1]
ranked_test_df = test_df.loc[:]

ranked_test_df['predictions'] = y_scores_final_test


ranked_test_df['rank'] = ranked_test_df.groupby('search_id')['predictions'].rank(ascending=False, method='first')


def calculate_ndcg(df, k=None):
    """Calculate normalized discounted cumulative gain at k for each group of search_id."""
    def dcg_at_k(rel, k):
        """Discounted cumulative gain at k."""
        rel = np.asfarray(rel)[:k] if k is not None else np.asfarray(rel)
        if rel.size:
            discounts = np.log2(np.arange(2, len(rel) + 2))
            return np.sum(rel / discounts)
        return 0

    def ndcg_at_k(rel_true, rel_pred, k):
        """Normalized discounted cumulative gain at k."""
        idcg = dcg_at_k(sorted(rel_true, reverse=True), k)
        if not idcg:
            return 0.
        return dcg_at_k(rel_pred, k) / idcg
    

    ndcg_scores = df.groupby('search_id').apply(
        lambda g: ndcg_at_k(
            g['target'].values,  
            g['predictions'].values, 
            k  
        )
    )
    return ndcg_scores

ndcg_scores = calculate_ndcg(ranked_test_df, k=5)  
overall_ndcg = ndcg_scores.mean() 
print(f"Overall nDCG score: {overall_ndcg}")

