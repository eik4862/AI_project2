from __future__ import annotations

import numpy as np
from scipy.stats import entropy, skew, kurtosis, ranksums
from scipy.interpolate import CubicSpline
from scipy.signal import welch, coherence
from scipy.integrate import simpson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Util import *


@final
class Preprocessor:
    def __init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def process(cls, src_dir: str, stim: str, *args: Any, normalize: bool = True,
                cache: bool = False) -> Dict[str, Any]:

        """
        Loads data and computes features.

        :param src_dir: Source directory.
        :param stim: Stimulus to be processed.
        :param args: Metadata for caching.
        :param normalize: If ture, it normalizes features.
        :param cache: It true, it uses cache.
        :return: Processed data.
        """
        # In case of cache hit, return it.
        if cache and Cache.lookup('_'.join(['data', *map(str, args)]) + '.pkl'):
            return Cache.load('_'.join(['data', *map(str, args)]) + '.pkl')

        # Load data.
        raw: Dict[str, Any] = cls.__collect(src_dir, stim)

        # Resample EEG.
        eeg_resampled: Dict[str, Any] = Preprocessor.__resample(raw['eeg'], 65)

        # Compute basic statistics: mean, variance, skewness, kurtosis.
        stats: Dict[str, Any] = cls.__stats(raw['eeg'], window=128, shift=2)
        stats_total: Dict[str, Any] = cls.__stats(raw['eeg'])

        # Compute bandpowers: delta, theta, beta, gamma, high gamma.
        bandpower: Dict[str, Any] = cls.__bandpower(raw['eeg'], window=128, shift=2)
        bandpower_total: Dict[str, Any] = cls.__bandpower(raw['eeg'])

        # Compute cross bandpowers: delta, theta, beta, gamma, high gamma.
        cross_bandpower: Dict[str, Any] = cls.__cross_bandpower(raw['eeg'], window=128, shift=2)
        cross_bandpower_total: Dict[str, Any] = cls.__cross_bandpower(raw['eeg'])

        # Compute entropy.
        _entropy: Dict[str, Any] = cls.__entropy(raw['eeg'], window=128, shift=2)
        entropy_total: Dict[str, Any] = cls.__entropy(raw['eeg'])

        # Wrap up.
        data_set: Dict[str, Any] = cls.__wrap_up(raw, eeg=eeg_resampled, stats=stats, bandpower=bandpower,
                                                 cross_bandpower=cross_bandpower, entropy=_entropy,
                                                 stats_total=stats_total, bandpower_total=bandpower_total,
                                                 cross_bandpower_total=cross_bandpower_total,
                                                 entropy_total=entropy_total)

        # Normalize.
        if normalize:
            data_set = cls.__normalize(data_set)

        # Write cache.
        if cache:
            Cache.save(data_set, '_'.join(['data', *map(str, args)]) + '.pkl')

        return data_set

    @classmethod
    def get_mask(cls, data_set: Dict[str, Any], pvalue: float = .05) -> Dict[str, np.ndarray]:
        """
        Generates masks for feature screening.
        It sets mask as true if a feature has no difference b/w two groups.

        :param data_set: Dataset whose masks is to be genereated.
        :param pvalue: Criterion p-value.
        :return: Dictionary holding masks.
        """
        mask_set: Dict[str, np.ndarray] = {}
        label: np.ndarray = data_set['label']

        for name, feature in data_set.items():
            if name in ['id', 'label'] or 'total' not in name:
                continue

            mask: List[bool] = []

            # Split features into two groups according to their labels.
            group_1, group_2 = feature['data'][label == 0], feature['data'][label == 1]
            progress: ProgressBar = ProgressBar(feature['data'].shape[1], 'feature',
                                                'Masking ' + '_'.join(name.split('_')[:-1]))

            for i in range(feature['data'].shape[1]):
                # If there is no difference, set mask as False.
                mask.append(ranksums(group_1[:, i], group_2[:, i])[1] < pvalue)
                progress.update()

            progress.close()
            mask_set[name] = np.array(mask)

        return mask_set

    @classmethod
    def screen(cls, data_set: Dict[str, Any], mask_set: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screens features.

        :param data_set: Dataset to be screened.
        :param mask_set: Mask used for screening.
        :return: Screened dataset.
        """
        for name, feature in data_set.items():
            if name in ['id', 'label', 'eeg']:
                continue

            # If mask is false, then drop that feature.
            if 'total' not in name:
                name += '_total'
                mask: np.ndarray = mask_set[name]
                feature['data'] = feature['data'][:, :, mask]
                feature['col_name'] = [feature['col_name'][i] for i in range(mask.shape[0]) if mask[i]]
            else:
                mask: np.ndarray = mask_set[name]
                feature['data'] = feature['data'][:, mask]
                feature['col_name'] = [feature['col_name'][i] for i in range(mask.shape[0]) if mask[i]]

        return data_set

    @classmethod
    def pca(cls, data_set: Dict[str, Any], threshold: float = .95, seed: Optional[int] = None) -> Dict[str, PCA]:
        """
        Runs principal component analysis and compute projection matrices.
        It keeps k principal components where k is the least # of principal components
        explaining threshold of total variance.

        :param data_set: Dataset to be analyzed.
        :param threshold: Threshold to determine the # of principal components to keep.
        :param seed: Seed for random process.
        :return: Dictionary holding PCA objects.
        """
        pca_set: Dict[str, PCA] = {}

        for name, feature in data_set.items():
            if name in ['id', 'label'] or 'total' not in name:
                continue

            # Run PCA, determine # of principal components to keep,
            # and then rerun PCA with the determined # of principal components.
            pca = PCA(random_state=seed).fit(feature['data'])
            n_comp: int = int(np.sum(np.cumsum(pca.explained_variance_ratio_) >= threshold))
            pca_set[name] = PCA(n_components=n_comp, random_state=seed).fit(feature['data'])

        return pca_set

    @classmethod
    def project(cls, data_set: Dict[str, Any], pca_set: Dict[str, PCA]) -> Dict[str, Any]:
        """
        Project feature.

        :param data_set: Dataset to be projected.
        :param pca_set: Set of PCA results holding projection matrices.
        :return: Projected dataset.
        """
        projected_set: Dict[str, Any] = {'id': data_set['id'], 'label': data_set['label'], 'eeg': data_set['eeg']}

        for name, feature in data_set.items():
            if name in ['id', 'label', 'eeg']:
                continue

            # Project features.
            if 'total' not in name:
                pca: PCA = pca_set[name + '_total']
                data: np.ndarray = feature['data']
                projected: List[np.ndarray] = []

                for i in range(data.shape[1]):
                    projected.append(pca.transform(data[:, i]))

                projected_set[name] = {'data': np.array(projected).swapaxes(0, 1), 'pca': pca}
            else:
                projected_set[name] = {'data': pca_set[name].transform(feature['data']), 'pca': pca}

        return projected_set

    @classmethod
    def split(cls, data_set: Dict[str, Any], ratio: float = .3,
              seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Splits dataset into train and test set.

        :param data_set: Dataset to be split.
        :param ratio: The ratio of test set size.
        :param seed: Seed for random process.
        :return: Train and test dataset.
        """
        train, test = {}, {}
        train_idx, test_idx = train_test_split(np.arange(data_set['label'].shape[0]), test_size=ratio,
                                               random_state=seed, shuffle=True)

        for name, feature in data_set.items():
            if name == 'label':
                train[name] = feature[train_idx]
                test[name] = feature[test_idx]
            elif name == 'id':
                train[name] = [feature[i] for i in train_idx]
                test[name] = [feature[i] for i in test_idx]
            else:
                train[name] = {'data': feature['data'][train_idx], 'col_name': feature['col_name']}
                test[name] = {'data': feature['data'][test_idx], 'col_name': feature['col_name']}

        return train, test

    @classmethod
    def extract(cls, data_set: Dict[str, Any], target: str) -> np.ndarray:
        """
        Extracts features from dataset.

        :param data_set: Dataset from which features are to be extracted.
        :param target: Target features.
        :return: Extracted features.
        """
        if target == 'total':
            augmented: List[np.ndarray] = []

            for name, feature in data_set.items():
                if 'total' in name:
                    augmented.append(feature['data'])

            return np.hstack(augmented)
        else:
            return data_set[target]['data']

    @classmethod
    def __collect(cls, src_dir: str, stim: str) -> Dict[str, Any]:
        """
        Load data.

        :param src_dir: Source directory.
        :param stim: Stimulus to be loaded.
        :return: Loaded data.
        """
        targets: List[str] = glob.glob(src_dir + '/*')
        data_set: Dict[str, Any] = {'label': [], 'id': [], 'eeg': {'col_name': [], 'data': []}}
        progress: ProgressBar = ProgressBar(len(targets), 'file', 'Collecting')

        for target in targets:
            with open(target, 'r') as f:
                lines: List[str] = list(f)

                # Skip erroneous or non-target stimulus trials.
                if any(['err' in line for line in lines]) or len(lines) < 10 or stim not in lines[3]:
                    progress.update()
                    continue

                # Parse data files.
                data_set['label'].append(1 if lines[0][5] == 'a' else 0)
                data_set['id'].append(lines[0].split()[1].split('.')[0])
                eeg: np.ndarray = np.zeros([256, 64])

                lines = lines[4:]
                i, j = -1, -1

                for line in lines:
                    if '#' in line:
                        if len(data_set['eeg']['col_name']) < 64:
                            data_set['eeg']['col_name'].append(line.split()[1])

                        i, j = 0, j + 1
                    else:
                        eeg[i, j] = float(line.split()[-1])
                        i += 1

                data_set['eeg']['data'].append(eeg)

            progress.update()

        progress.close()

        data_set['label'] = np.array(data_set['label'])
        data_set['eeg']['data'] = np.array(data_set['eeg']['data'])

        return data_set

    @classmethod
    def __resample(cls, eeg: Dict[str, Any], n_resample: int) -> Dict[str, Any]:
        """

        :param eeg: EEG data.
        :param n_resample: Size of resampled EEG.
        :return: Resampled EEG data.
        """
        result: List[np.ndarray] = []
        col_name: List[str] = eeg['col_name']
        data: np.ndarray = eeg['data']
        t: np.ndarray = np.arange(eeg['data'].shape[1])
        t_resample = np.linspace(0, len(t), num=n_resample, endpoint=False)
        progress: ProgressBar = ProgressBar(data.shape[0], 'file', 'Resampling')

        for sample in data:
            result.append(np.array([CubicSpline(t, sample[:, i])(t_resample)
                                    for i in range(sample.shape[1])]).transpose())
            progress.update()

        progress.close()

        return {'data': np.array(result), 'col_name': col_name}

    @classmethod
    def __stats(cls, eeg: Dict[str, Any], window: Optional[int] = None, shift: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes basic statistics: mean, variance, skewness, kurtosis.
        If window width or shift amount is not given, it computes statistics in whole time range.

        :param eeg: EEG data.
        :param window: Window width.
        :param shift: Shift amount of window.
        :return: Dictionary holding computed statistics and error flags.
        """
        result: List[np.ndarray] = []
        shifted: np.ndarray = cls.__shift_eeg(eeg, window, shift)
        col_name: List[str] = [name + '_' + stat for stat in ['mean', 'variance', 'skewness', 'kurtosis']
                               for name in eeg['col_name']]
        error_flag: List[bool] = []
        progress: ProgressBar = ProgressBar(shifted.shape[0], 'sample', 'Computing basic stats')

        for sample in shifted:
            # Compute basic statistics.
            result.append(np.hstack([np.mean(sample, axis=1), np.var(sample, axis=1), skew(sample, axis=1),
                                     kurtosis(sample, axis=1)]).squeeze())

            # Check for errors.
            error_flag.append(cls.__check_error(result[-1]))
            progress.update()

        progress.close()

        return {'data': np.array(result), 'col_name': col_name, 'error_flag': np.array(error_flag)}

    @classmethod
    def __bandpower(cls, eeg: Dict[str, Any], window: Optional[int] = None,
                    shift: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes bandpowers: delta, theta, beta, gamma, high gamma.
        If window width or shift amount is not given, it computes bandpowers in whole time range.

        :param eeg: EEG data.
        :param window: Window width.
        :param shift: Shift amount of window.
        :return: Dictionary holding computed bandpowers and error flags.
        """
        result: List[np.ndarray] = []
        shifted: np.ndarray = cls.__shift_eeg(eeg, window, shift)
        col_name: List[str] = [name + '_' + band for band in ['delta', 'theta', 'beta', 'gamma', 'high_gamma']
                               for name in eeg['col_name']]
        error_flag: List[bool] = []
        progress: ProgressBar = ProgressBar(shifted.shape[0], 'sample', 'Computing bandpower')

        for sample in shifted:
            # For each time window, estimate PSD and compute bandpowers.
            freq, Pxx = welch(sample, fs=256, nperseg=128, axis=1)
            result.append(np.hstack([cls.__delta(Pxx, freq), cls.__theta(Pxx, freq), cls.__beta(Pxx, freq),
                                     cls.__gamma(Pxx, freq), cls.__high_gamma(Pxx, freq)]).squeeze())

            # Check for errors.
            error_flag.append(cls.__check_error(result[-1]))
            progress.update()

        progress.close()

        return {'data': np.array(result), 'error_flag': np.array(error_flag), 'col_name': col_name}

    @classmethod
    def __cross_bandpower(cls, eeg: Dict[str, Any], window: Optional[int] = None,
                          shift: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes cross bandpowers: delta, theta, beta, gamma, high gamma.
        If window width or shift amount is not given, it computes bandpowers in whole time range.

        :param eeg: EEG data.
        :param window: Window width.
        :param shift: Shift amount of window.
        :return: Dictionary holding computed cross bandpowers and error flags.
        """
        result: List[np.ndarray] = []
        permutation_names: List[str] = []
        error_flag: List[bool] = []

        for i in range(eeg['data'].shape[2]):
            for j in range(eeg['data'].shape[2]):
                if j <= i:
                    continue
                permutation_names.append(eeg['col_name'][i] + '_' + eeg['col_name'][j])

        col_name: List[str] = [name + '_' + band for band in ['delta', 'theta', 'beta', 'gamma', 'high_gamma']
                               for name in permutation_names]
        progress: ProgressBar = ProgressBar(eeg['data'].shape[0], 'sample', 'Computing cross bandpower')

        for sample in eeg['data']:
            permutation_1, permutation_2 = [], []

            for i in range(sample.shape[1]):
                for j in range(sample.shape[1]):
                    if j <= i:
                        continue

                    permutation_1.append(sample[:, i])
                    permutation_2.append(sample[:, j])

            permutation_1, permutation_2 = np.array(permutation_1).transpose(), np.array(permutation_2).transpose()
            shifted_1: np.ndarray = cls.__shift_eeg({'data': np.array([permutation_1])}, window, shift)[0]
            shifted_2: np.ndarray = cls.__shift_eeg({'data': np.array([permutation_2])}, window, shift)[0]

            # For each time window, estimate CSD and compute cross bandpowers.
            with np.errstate(divide='ignore', invalid='ignore'):
                freq, Cxy = coherence(shifted_1, shifted_2, fs=256, nperseg=128, axis=1)

            result.append(np.hstack([cls.__delta(Cxy, freq), cls.__theta(Cxy, freq), cls.__beta(Cxy, freq),
                                     cls.__gamma(Cxy, freq), cls.__high_gamma(Cxy, freq)]).squeeze())

            # Check for errors.
            error_flag.append(cls.__check_error(result[-1]))
            progress.update()

        progress.close()

        return {'data': np.array(result), 'error_flag': np.array(error_flag), 'col_name': col_name}

    @classmethod
    def __entropy(cls, eeg: Dict[str, Any], window: Optional[int] = None,
                  shift: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes entropy.
        If window width or shift amount is not given, it computes entropy in whole time range.

        :param eeg: Dataset.
        :param window: Window width.
        :param shift: Shift amount of window.
        :return: Dictionary holding computed entropies and error flags.
        """
        result: List[np.ndarray] = []
        shifted: np.ndarray = cls.__shift_eeg(eeg, window, shift)
        col_name: List[str] = [name + '_entropy' for name in eeg['col_name']]
        error_flag: List[bool] = []
        progress: ProgressBar = ProgressBar(shifted.shape[0], 'sample', 'Computing entropy')

        for sample in shifted:
            # For each time window, estimate PSD and compute entropy.
            Pxx: np.ndarray = welch(sample, fs=256, nperseg=128, axis=1)[1]

            with np.errstate(divide='ignore', invalid='ignore'):
                result.append(entropy(Pxx, axis=1).squeeze())

            # Check for errors.
            error_flag.append(cls.__check_error(result[-1]))
            progress.update()

        progress.close()

        return {'data': np.array(result), 'error_flag': np.array(error_flag), 'col_name': col_name}

    @classmethod
    def __wrap_up(cls, raw: Dict[str, Any], **kwagrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Groups computed features excluding erroneous ones.

        :param raw: Raw dataset.
        :param kwargs: Features to be grouped.
        :return: Grouped features.
        """
        result: Dict[str, Any] = {'label': raw['label'], 'id': raw['id']}

        error_flag: np.ndarray = np.repeat(False, raw['eeg']['data'].shape[0])

        for _, feature in kwagrs.items():
            if 'error_flag' in feature.keys():
                error_flag = np.logical_or(error_flag, feature['error_flag'])

        result['label'] = result['label'][np.logical_not(error_flag)]
        result['id'] = [result['id'][i] for i in range(error_flag.shape[0]) if not error_flag[i]]

        for name, feature in kwagrs.items():
            result[name] = {'data': feature['data'][np.logical_not(error_flag)], 'col_name': feature['col_name']}

        return result

    @classmethod
    def __normalize(cls, data_set: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes features.

        :param data_set: Dataset to be normalized.
        :return: Normalized features.
        """
        for name, feature in data_set.items():
            if name in ['label', 'id']:
                continue

            data: np.ndarray = feature['data']

            if data.ndim == 3:
                # Dimension of data is 3, reshape it first, normalize, and then reshape it back.
                stacked: np.ndarray = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
                feature['data'] = StandardScaler().fit_transform(stacked).reshape(data.shape)
            else:
                feature['data'] = StandardScaler().fit_transform(data)

        return data_set

    @classmethod
    def __shift_eeg(cls, eeg: Dict[str, Any], window: Optional[int] = None,
                    shift: Optional[int] = None) -> np.ndarray:
        if window is None or shift is None:
            return np.array([eeg['data']]).swapaxes(0, 1)
        else:
            i: int = 0
            data: np.ndarray = eeg['data']
            shifted: List[np.ndarray] = []

            while shift * i + window <= data.shape[1]:
                shifted.append(data[:, (shift * i):((shift * i) + window)])
                i += 1

            return np.array(shifted).swapaxes(0, 1)

    @classmethod
    def __check_error(cls, feature: np.ndarray) -> bool:
        return np.isnan(feature).any() or np.isinf(feature).any()

    @classmethod
    def __delta(cls, psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return simpson(psd[:, freq <= 4], dx=freq[1] - freq[0], axis=1)

    @classmethod
    def __theta(cls, psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return simpson(psd[:, np.logical_and(4 < freq, freq <= 12)], dx=freq[1] - freq[0], axis=1)

    @classmethod
    def __beta(cls, psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return simpson(psd[:, np.logical_and(12 < freq, freq <= 30)], dx=freq[1] - freq[0], axis=1)

    @classmethod
    def __gamma(cls, psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return simpson(psd[:, np.logical_and(30 < freq, freq <= 50)], dx=freq[1] - freq[0], axis=1)

    @classmethod
    def __high_gamma(cls, psd: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return simpson(psd[:, np.logical_and(50 < freq, freq <= 100)], dx=freq[1] - freq[0], axis=1)
