from __future__ import annotations

from sklearn.svm import SVC
from Classifier import *
from Util import *


@final
class SVM(Classifier):
    """
    Wrapper for support vector machine classifier.
    """

    def __init__(self, C: float = 1, gamma: float = None, seed: Optional[int] = None) -> None:
        """
        :param C: Amount of penalty.
        :param gamma: Dispersion of Gaussian kernel.
        :param seed: Seed for random process.
        """
        self.__C: float = C
        self.__gamma: float = gamma
        self.__seed: Optional[int] = seed
        self.__unit: Optional[SVC] = None

    @classmethod
    def _get_param_names(cls) -> List[str]:
        return ['C', 'gamma']

    def get_name(self) -> str:
        return 'HMM_SVM'

    def get_params(self, deep=True) -> Dict[str, float]:
        return {'C': self.__C, 'gamma': self.__gamma}

    def set_params(self, **params: Dict[str, float]) -> SVM:
        self.__C = params['C']
        self.__gamma = params['gamma']

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, cache: bool = False) -> SVM:
        """
        Fits SVM.

        :param X: Train data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param cache: If true, it uses cache.
        :return: Fitted self.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['fit', 'HMM_SVM', *map(str, args)]) + '.pkl'):
            self.__unit = Cache.load('_'.join(['fit', 'HMM_SVM', *map(str, args)]) + '.pkl')

            return self

        # Fit SVM.
        self.__unit = SVC(C=self.__C, kernel='rbf', gamma=self.__gamma, probability=True,
                          random_state=self.__seed).fit(X, y)

        # Write cache.
        if cache:
            Cache.save(self.__unit, '_'.join(['fit', 'HMM_SVM', *map(str, args)]) + '.pkl')

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
