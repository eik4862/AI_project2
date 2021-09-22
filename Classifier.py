from __future__ import annotations

import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import BaseEstimator
from sklearn.model_selection import HalvingGridSearchCV
from sklearn import metrics
from Util import *


class Classifier(BaseEstimator):
    """
    Base class for classifiers.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, *args) -> Classifier:
        """
        Fits classifier.

        :param X: Train data.
        :param y: True labels.
        :param args: Additional arguments.
        :return: Fitted self.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts new data.

        :param X: New data.
        :return: Predicted labels.
        """
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Computes scores of new data.

        :param X: New data.
        :return: Computed scores.
        """
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    @classmethod
    def grid_search(cls, classifier: Classifier, parameters: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray,
                    *args: Any, seed: Optional[int] = None, fold: int = 5, cache: bool = False, csv: bool = False,
                    verbose: bool = True) -> HalvingGridSearchCV:
        """
        Grid search for hyperparameter tuning.
        It uses K-fold CV and F1 score as metric for tuning.

        :param classifier: Classifier to be tuned.
        :param parameters: Candidate parameters.
        :param X: Train data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param seed: Seed for random process.
        :param fold: # of folds.
        :param cache: If true, it uses cache.
        :param csv: If true, it exports the result as CSV.
        :param verbose: If true, it prints out summary.
        :return: HalvingGridSearchCV object holding tuning results.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['tune', classifier.get_name(), *map(str, args)]) + '.pkl'):
            result: HalvingGridSearchCV = Cache.load(
                '_'.join(['tune', classifier.get_name(), *map(str, args)]) + '.pkl')

            if verbose:
                print(f'@selected: {result.best_params_}')
                print(f'@score   : {result.best_score_}')
                print()

            return result

        # Perform CV.
        result: HalvingGridSearchCV = HalvingGridSearchCV(classifier, parameters, scoring='accuracy', n_jobs=-1,
                                                          refit=False, cv=fold, verbose=3 if verbose else 0,
                                                          random_state=seed).fit(X, y)

        # Write CSV.
        if csv:
            param_names: List[str] = list(parameters.keys())
            header: List[str] = param_names + ['acc.mean', 'acc.sd']
            body: List[List[Any]] = np.vstack([np.array(result.cv_results_['param_' + name]) for name in param_names]
                                              + [result.cv_results_['mean_test_score'],
                                                 result.cv_results_['std_test_score']]).transpose().tolist()
            Writer.write('_'.join(['tune', classifier.get_name(), *map(str, args)]) + '.csv', header, body)

        # Write cache.
        if cache:
            Cache.save(result, '_'.join(['tune', classifier.get_name(), *map(str, args)]) + '.pkl')

        # Print out the result.
        if verbose:
            print(f'@selected: {result.best_params_}')
            print(f'@score   : {result.best_score_}')
            print()

        return result

    @classmethod
    def test(cls, classifier: Classifier, X: np.ndarray, y: np.ndarray, *args: Any, cache: bool = False,
             csv: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """
        Test classifier.
        The classifier must be fitted before using this method.

        :param classifier: Classifier to be tested.
        :param X: Test data.
        :param y: True labels.
        :param args: Metadata for caching.
        :param cache: If true, it uses cache.
        :param csv: If true, it exports the result as CSV.
        :param verbose: If true, it prints out summary.
        :return: Dictionary holding test results.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['test', classifier.get_name(), *map(str, args)]) + '.pkl'):
            result: Dict[str, Any] = Cache.load('_'.join(['test', classifier.get_name(), *map(str, args)]) + '.pkl')
            stats, confusion = result['stats'], result['confusion']

            if verbose:
                for stat_name in ['tpr', 'ppv', 'acc', 'f1', 'kappa', 'mcc', 'auc']:
                    print(f'@{stat_name.ljust(5)}: {round(stats[stat_name], 4)}')

                print('@confusion matrix')

                for row in confusion.tolist():
                    print('  ' + ' '.join(map(str, row)))

                print()

            return result

        # Predicts and compute scores.
        y_pred, y_score = classifier.predict(X), classifier.score(X)[:, -1]

        # Compute various statistics.
        confusion: np.ndarray = metrics.confusion_matrix(y, y_pred)
        tpr, ppv = metrics.recall_score(y, y_pred), metrics.precision_score(y, y_pred)
        acc, f1 = metrics.accuracy_score(y, y_pred), metrics.f1_score(y, y_pred)
        kappa, mcc = metrics.cohen_kappa_score(y, y_pred), metrics.matthews_corrcoef(y, y_pred)

        roc: Tuple[np.ndarray] = metrics.roc_curve(y, y_score)
        roc: Dict[str, np.ndarray] = {'fpr': roc[0], 'tpr': roc[1]}
        auc: float = metrics.roc_auc_score(y, y_score)

        stats: Dict[str, float] = {'tpr': tpr, 'ppv': ppv, 'acc': acc, 'f1': f1, 'kappa': kappa, 'mcc': mcc, 'auc': auc}
        result: Dict[str, Any] = {'stats': stats, 'roc': roc, 'confusion': confusion, 'label': y_pred, 'score': y_score}

        # Write CSV.
        if csv:
            header: List[str] = ['fpr', 'tpr']
            body: List[List[Any]] = np.vstack([roc['fpr'], roc['tpr']]).transpose().tolist()

            Writer.write('_'.join(['roc', classifier.get_name(), *map(str, args)]) + '.csv', header, body)

        # Write cache.
        if cache:
            Cache.save(result, '_'.join(['test', classifier.get_name(), *map(str, args)]) + '.pkl')

        # Print out the result.
        if verbose:
            for stat_name in ['tpr', 'ppv', 'acc', 'f1', 'kappa', 'mcc', 'auc']:
                print(f'@{stat_name.ljust(5)}: {round(stats[stat_name], 4)}')

            print('@confusion matrix')

            for row in confusion.tolist():
                print('  ' + ' '.join(map(str, row)))

            print()

        return result
