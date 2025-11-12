import numpy as np
from scipy.stats import mode 

class HybridEnsembleModel:
    def __init__(self, svm, nn, xgb):
        self.svm = svm
        self.nn = nn
        self.xgb = xgb

    def predict(self, X):
        svm_pred = self.svm.predict(X)
        nn_pred = self.nn.predict(X)
        xgb_pred = self.xgb.predict(X)

        stacked = np.vstack([svm_pred, nn_pred, xgb_pred])
        majority_vote, _ = mode(stacked, axis=0)
        return majority_vote.flatten()