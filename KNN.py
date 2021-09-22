from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from Classifier import *
from Util import *


@final
class KNN(Classifier):
    """
    Wrapper for k-nearest neighbor classifier.
    """
    def __init__(self, k: int = 5) -> None:
        self.__k: int = k
        self.__unit: Optional[KNeighborsClassifier] = None

    @classmethod
    def _get_param_names(cls) -> List[str]:
        return ['k']

    def get_name(self) -> str:
        return 'KNN'

    def get_params(self, deep=True) -> Dict[str, int]:
        return {'k': self.__k}

    def set_params(self, **params: int) -> KNN:
        self.__k = params['k']

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, cache: bool = False) -> KNN:
        """
        Fits KNN.

        :param X: Train data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param cache: If true, it uses cache.
        :return: Fitted self.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['fit', 'KNN', *map(str, args)]) + '.pkl'):
            self.__unit = Cache.load('_'.join(['fit', 'KNN', *map(str, args)]) + '.pkl')

            return self

        # Fits KNN.
        self.__unit = KNeighborsClassifier(n_neighbors=self.__k).fit(X, y)

        # Write cache.
        if cache:
            Cache.save(self.__unit, '_'.join(['fit', 'KNN', *map(str, args)]) + '.pkl')

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
