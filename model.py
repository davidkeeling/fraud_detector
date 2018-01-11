import pandas as pd
import _pickle as pickle
from bs4 import BeautifulSoup
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


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
        self._strip_html(X)
        X = X['text_desc']
        tfidf = self._vectorizer.fit_transform(X)
        topics = self._topic_modeler.fit_transform(tfidf)
        self._classifier.fit(topics, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        self._strip_html(X)
        X = X['text_desc']
        tfidf = self._vectorizer.transform(X)
        topics = self._topic_modeler.transform(tfidf)
        return self._classifier.predict_proba(topics)

    def predict(self, X):
        """Make predictions on new data."""
        self._strip_html(X)
        X = X['text_desc']
        tfidf = self._vectorizer.transform(X)
        topics = self._topic_modeler.transform(tfidf)
        return self._classifier.predict(topics)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        prob_predictions = self.predict_proba(X)
        auc = roc_auc_score(y, prob_predictions[:, 1])
        print('Area under ROC curve: ', auc)
        tn, fp, fn, tp = confusion_matrix(y, self.predict(X)).ravel()
        total = len(y)
        print("""
        True negatives (detected nonfraud): {} ({}%)
        False positive (falsely labelled as fraud): {} ({}%)
        False negatives (failed to detect fraud): {} ({}%)
        True positives (detected fraud!): {} ({}%)
        """.format(tn, 100*tn/total, fp, 100*fp/total, fn, 100*fn/total, tp, 100*tp/total))
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
    X, y = get_data("data/data.json")
    fd = FraudDetector()
    fd.fit(X, y)
    print(fd.predict(X))

    with open('data/model.pkl', 'wb') as f:
        pickle.dump(fd, f)
