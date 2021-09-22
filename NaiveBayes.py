from __future__ import annotations

from sklearn.naive_bayes import GaussianNB
from Classifier import *
from Util import *


@final
class NaiveBayes(Classifier):
    """
    Wrapper for naive Bayes classifier.
    """
    def __init__(self) -> None:
        self.__unit: Optional[GaussianNB] = None

    @classmethod
    def _get_param_names(cls) -> List[str]:
        return []

    def get_name(self) -> str:
        return 'NB'

    def get_params(self, deep=True) -> Dict[str, int]:
        return {}

    def set_params(self, **params: int) -> NaiveBayes:
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, cache: bool = False) -> NaiveBayes:
        """
        Fits naive Bayes classifier.

        :param X: Train data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param cache: If true, it uses cache.
        :return: Fitted self.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['fit', 'NB', *map(str, args)]) + '.pkl'):
            self.__unit = Cache.load('_'.join(['fit', 'NB', *map(str, args)]) + '.pkl')

            return self

        # Fits naive Bayes classifier.
        self.__unit = GaussianNB().fit(X, y)

        # Write cache.
        if cache:
            Cache.save(self.__unit, '_'.join(['fit', 'NB', *map(str, args)]) + '.pkl')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts new data.

        :param X: New data.
        :return: Predicted labels.
        """
        if not self.__unit:
            raise NotImplementedError

        return self.__unit.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Computes probability of belonging each label.

        :param X: New data.
        :return: Computed probabilities.
        """
        if not self.__unit:
            raise NotImplementedError

        return self.__unit.predict_proba(X)
