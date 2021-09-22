from __future__ import annotations

from hmmlearn import hmm
from Classifier import *
from Util import *


@final
class HMM(Classifier):
    """
    Classifier using hidden Markov model.
    """

    def __init__(self, k: int = 1, seed: Optional[int] = None) -> None:
        """
        :param k: # of hidden states.
        :param seed: Seed for random process.
        """
        self.__k: int = k
        self.__seed: Optional[int] = seed
        self.__units: Optional[List[hmm.GaussianHMM]] = None
        self.__labels: Optional[List[int]] = None

    @classmethod
    def _get_param_names(cls) -> List[str]:
        return ['k']

    def get_name(self) -> str:
        return 'HMM'

    def get_params(self, deep=True) -> Dict[str, int]:
        return {'k': self.__k}

    def set_params(self, **params: int) -> HMM:
        self.__k = params['k']

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, max_iter: int = 10, cache: bool = False,
            verbose: bool = False) -> HMM:
        """
        Fits HMM.

        :param X: Train data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param max_iter: # of max iteration.
        :param cache: If true, it uses cache.
        :param verbose: If true, it prints out messages during fitting.
        :return: Fitted self.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['fit', 'HMM', *map(str, args)]) + '.pkl'):
            self.__units, self.__labels = Cache.load('_'.join(['fit', 'HMM', *map(str, args)]) + '.pkl')

            return self

        self.__units = []
        self.__labels = sorted(list({*y}))

        # Split data according to the label.
        X_split: List[np.ndarray] = [X[y == label] for label in self.__labels]

        # Fit one Gaussian HMM per each label.
        for X in X_split:
            unit = hmm.GaussianHMM(n_components=self.__k, covariance_type='diag', n_iter=max_iter,
                                   random_state=self.__seed, verbose=verbose)
            unit.fit(X.reshape([X.shape[0] * X.shape[1], X.shape[2]]), np.repeat(X.shape[1], X.shape[0]))
            self.__units.append(unit)

        # Write cache.
        if cache:
            Cache.save((self.__units, self.__labels), '_'.join(['fit', 'HMM', *map(str, args)]) + '.pkl')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts new data.

        :param X: New data.
        :return: Predicted labels.
        """
        if not self.__units:
            raise NotImplementedError

        y_pred: np.ndarray = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            scores: np.ndarray = np.array([unit.score(X[i]) for unit in self.__units])
            y_pred[i] = self.__labels[np.nanargmax(scores)]

        return y_pred

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Computes log likelihoods of new data.

        :param X: New data.
        :return: Computed log like
        """
        if not self.__units:
            raise NotImplementedError

        y_score: np.ndarray = np.zeros([X.shape[0], len(self.__labels)])

        for i in range(X.shape[0]):
            y_score[i] = np.array([unit.score(X[i]) for unit in self.__units])

        return y_score
