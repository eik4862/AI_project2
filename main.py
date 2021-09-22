from __future__ import annotations

from Preprocessor import *
from HMM import *
from SVM import *
from NaiveBayes import *
from KNN import *

def compute_scores(hmms: Dict[str, HMM], data_set: Dict[str, Any], *args: Any, cache: bool = False) -> np.ndarray:
    """
    Compute scores using HMMs.

    :param hmms: HMMs to use.
    :param data_set: Dataset.
    :param args: Metadata for caching.
    :param cache: If true, it uses cache.
    :return: Computed scores.
    """
    if cache and Cache.lookup('_'.join(['score', *map(str, args)]) + '.pkl'):
        return Cache.load('_'.join(['score', *map(str, args)]) + '.pkl')

    scores: np.ndarray = np.hstack([hmm.score(data_set[name]['data']) for name, hmm in hmms.items()])
    scores = StandardScaler().fit_transform(scores)

    if cache:
        Cache.save(scores, '_'.join(['score', *map(str, args)]) + '.pkl')

    return scores


if __name__ == '__main__':
    ### PREPROCESS
    # WARNING: Code below show be run only ONCE.
    # batch: Dict[str, Any] = Preprocessor.process('Batch', 'S1', 'batch', cache=True)
    # full: Dict[str, Any] = Preprocessor.process('Data', 'S1', 'full', cache=True)
    # mask: Dict[str, np.ndarray] = Preprocessor.get_mask(batch)
    # Cache.save(Preprocessor.screen(batch, mask), 'data_batch.pkl')
    # Cache.save(Preprocessor.split(Preprocessor.screen(full, mask)), 'data_full.pkl')
    # pca: Dict[str, PCA] = Preprocessor.pca(batch, seed=1)
    # Cache.save(Preprocessor.project(batch, pca), 'data_batch_projected.pkl')
    # Cache.save((Preprocessor.project(train, pca), Preprocessor.project(test, pca)), 'data_full_projected.pkl')

    ### PARAMETER TUNING
    batch: Dict[str, Any] = Cache.load('data_batch_projected.pkl')

    # HMM
    parameters: Dict[str, np.ndarray] = {'k': np.linspace(1, 20, num=20, dtype=int)}
    for name in ['eeg', 'stats', 'bandpower', 'cross_bandpower', 'entropy']:
        Classifier.grid_search(HMM(seed=1), parameters, batch[name]['data'], batch['label'], name, seed=1, cache=True,
                               csv=True)

    # HMM + SVM
    hmms: Dict[str, HMM] = {'eeg': HMM(k=16, seed=1), 'stats': HMM(k=14, seed=1), 'bandpower': HMM(k=18, seed=1),
                            'cross_bandpower': HMM(k=11, seed=1), 'entropy': HMM(k=10, seed=1)}
    for name in ['eeg', 'stats', 'bandpower', 'cross_bandpower', 'entropy']:
        hmms[name].fit(batch[name]['data'], batch['label'], 'tune', name, verbose=True, cache=True)
    batch_score: np.ndarray = compute_scores(hmms, batch, 'batch', cache=True)
    parameters: Dict[str, np.ndarray] = {'C': 10 ** np.arange(-3, 9, dtype=float),
                                         'gamma': 10 ** np.arange(-9, 3, dtype=float)}
    Classifier.grid_search(SVM(seed=1), parameters, np.hstack([batch_score, Preprocessor.extract(batch, 'total')]),
                           batch['label'], seed=1, cache=True, csv=True)

    # KNN
    parameters: Dict[str, np.ndarray] = {'k': np.linspace(1, 20, num=20, dtype=int)}
    Classifier.grid_search(KNN(), parameters, Preprocessor.extract(batch, 'total'), batch['label'], seed=1, cache=True,
                           csv=True)

    ### FITTING
    train, test = Cache.load('data_full_projected.pkl')

    # HMM
    hmms: Dict[str, HMM] = {'eeg': HMM(k=16, seed=1), 'stats': HMM(k=14, seed=1), 'bandpower': HMM(k=18, seed=1),
                            'cross_bandpower': HMM(k=11, seed=1), 'entropy': HMM(k=10, seed=1)}
    for name in ['eeg', 'stats', 'bandpower', 'cross_bandpower', 'entropy']:
        hmms[name].fit(train[name]['data'], train['label'], name, max_iter=100, verbose=True, cache=True)

    # HMM + SVM
    train_score: np.ndarray = compute_scores(hmms, train, 'train', cache=True)
    hmm_svm: SVM = SVM(C=1000, gamma=.001, seed=1).fit(np.hstack([train_score, Preprocessor.extract(train, 'total')]),
                                                       train['label'], cache=True)

    # Naive Bayes
    naive_bayes: NaiveBayes = NaiveBayes().fit(Preprocessor.extract(train, 'total'), train['label'], cache=True)

    # KNN
    knn: KNN = KNN(k=2).fit(Preprocessor.extract(train, 'total'), train['label'], cache=True)

    ### TESTING
    results: Dict[str, Dict[str, Any]] = {}

    # HMM
    for name in ['eeg', 'stats', 'bandpower', 'cross_bandpower', 'entropy']:
        results['HMM_' + name] = Classifier.test(hmms[name], test[name]['data'], test['label'], name, cache=True,
                                                 csv=True, verbose=True)

    # HMM + SVM
    test_score: np.ndarray = compute_scores(hmms, test, 'test', cache=True)
    results['HMM_SVM'] = Classifier.test(hmm_svm, np.hstack([test_score, Preprocessor.extract(test, 'total')]),
                                         test['label'], cache=True, csv=True, verbose=True)

    # Naive Bayes
    results['NB'] = Classifier.test(naive_bayes, Preprocessor.extract(test, 'total'), test['label'], cache=True,
                                    csv=True, verbose=True)

    # KNN
    results['KNN'] = Classifier.test(knn, Preprocessor.extract(test, 'total'), test['label'], cache=True, csv=True,
                                     verbose=True)

    # Export test results.
    header = ['classifier', 'tpr', 'ppv', 'acc', 'f1', 'kappa', 'mcc', 'auc']
    body = []

    for classifier_name, result in results.items():
        body.append([classifier_name, result['stats']['tpr'], result['stats']['ppv'], result['stats']['acc'],
                     result['stats']['f1'], result['stats']['kappa'], result['stats']['mcc'], result['stats']['auc']])

    Writer.write('summary.csv', header, body)
