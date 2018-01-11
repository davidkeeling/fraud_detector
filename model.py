import numpy as np
import pandas as pd
import _pickle as pickle
from bs4 import BeautifulSoup
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


class FraudDetector(object):
    """A classifier model:
        - Vectorize the raw text from data point description into features.
        - Fit classifier to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._topic_modeler = LatentDirichletAllocation()
        self._classifier = RandomForestClassifier()

    def _strip_html(self, df):
        df['text_desc'] = [self._strip_html_row(r['description']) for ind, r in df.iterrows()]

    def _strip_html_row(self, desc):
        soup = BeautifulSoup(desc, 'html.parser')
        return soup.get_text()

    def _feature_engineer(self, df, training=False):
        df['event_created'] = pd.to_datetime(df['event_created'],unit='s')
        df['user_created'] = pd.to_datetime(df['user_created'],unit='s')
        #add column that shows length of time user had an account before event was created
        df['user_account_length'] = df['event_created'] - df['user_created']
        #only use the days from this difference
        df['user_account_length'] = pd.DatetimeIndex(df['user_account_length']).day

        #adds a column showing whether the event has an organization description
        df['org_desc_cleaned'] = df['org_desc'].astype(str)
        mask = (df['org_desc_cleaned'].str.len() == 0)
        df['is_org_desc'] = np.where(mask, 1, 0)

        df_final = df[['description', 'body_length', 'user_type', 'user_age', 'user_account_length', 'is_org_desc',
                        'num_payouts', 'name_length', 'sale_duration2',
                        'num_order']]

        scale_list = ['body_length', 'user_age', 'user_account_length', 'num_payouts',
                        'name_length', 'sale_duration2','num_order']
        if training:
            self._scaler = StandardScaler()
            df_final.loc[:, scale_list] = self._scaler.fit_transform(df_final[scale_list])
        else:
            df_final.loc[:, scale_list] = self._scaler.transform(df_final[scale_list])
        return df_final

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        pattern = r'fraud'
        mask = X['acct_type'].str.contains(pattern)
        y = np.where(mask, 1, 0)

        X = self._feature_engineer(X, training=True)
        self._strip_html(X)
        tfidf = self._vectorizer.fit_transform(X['text_desc'])
        topics = self._topic_modeler.fit_transform(tfidf)
        topics = pd.DataFrame(topics)
        X = pd.concat([X.reset_index(), topics], axis=1)
        X.drop(['description', 'text_desc'], axis=1,inplace=True)
        self._classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        X = self._feature_engineer(X, training=False)
        self._strip_html(X)
        tfidf = self._vectorizer.transform(X['text_desc'])
        topics = self._topic_modeler.transform(tfidf)
        topics = pd.DataFrame(topics)
        X = pd.concat([X.reset_index(), topics], axis=1)
        X.drop(['description', 'text_desc'], inplace=True, axis=1)
        return self._classifier.predict_proba(X)

    def predict(self, X):
        """Make predictions on new data."""
        X = self._feature_engineer(X, training=False)
        self._strip_html(X)
        tfidf = self._vectorizer.transform(X['text_desc'])
        topics = self._topic_modeler.transform(tfidf)
        topics = pd.DataFrame(topics)
        X = pd.concat([X.reset_index(), topics], axis=1)
        X.drop(['description', 'text_desc'], inplace=True, axis=1)
        return self._classifier.predict(X)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        prob_predictions = self.predict_proba(X)
        auc = roc_auc_score(y, prob_predictions[:, 1])
        print('Area under ROC curve: ', auc)
        y_pred = self.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        total = len(y)
        print("""
True negatives (detected nonfraud): {} ({}%)
False positive (falsely labelled as fraud): {} ({}%)
False negatives (failed to detect fraud): {} ({}%)
True positives (detected fraud!): {} ({}%)
F1 score: {}
Recall: {}
Precision: {}
        """.format(tn, 100*tn/total, fp, 100*fp/total, fn, 100*fn/total, tp, 100*tp/total, f1, recall, precision))
        return auc

def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    data = pd.read_json(filename)
    df_fraud = data[data['acct_type'].str.startswith('fraud')]
    df_notfraud = data[data['acct_type'].str.startswith('fraud') == False]
    n_notfraud = len(df_notfraud)
    df_fraud_upsampled = resample(df_fraud, replace=True, n_samples=n_notfraud, random_state=0)
    data = pd.concat([df_notfraud, df_fraud_upsampled])
    y = data['acct_type'].str.startswith('fraud')
    X = data.drop('acct_type')
    return X, y


if __name__ == '__main__':
    # X, y = get_data("data/data.json")
    X, y = get_data("data/data.json")
    fd = FraudDetector()
    fd.fit(X, y)

    with open('data/model.pkl', 'wb') as f:
        pickle.dump(fd, f)
