from __future__ import annotations

from typing import *
import glob
import pickle
import csv
from tqdm import tqdm


@final
class Writer:
    """
    CSV writer.
    """

    @classmethod
    def write(cls, fname: str, header: List[str], body: List[List[Any]]) -> None:
        with open('./Exports/' + fname, 'w', newline='') as f:
            writer: csv.writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(body)


@final
class ProgressBar:
    """
    Progress bar.
    """

    def __init__(self, total: int, unit: str, task: str) -> None:
        self.__progress: tqdm = tqdm(total=total, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', unit=unit, desc=task)

    def update(self) -> None:
        self.__progress.update()

    def close(self) -> None:
        self.__progress.close()


@final
class Cache:
    """
    Cache memory.
    """

    @classmethod
    def lookup(cls, pkl_name: str) -> bool:
        return 'Cache/' + pkl_name in glob.glob('Cache/*.pkl')

    @classmethod
    def load(cls, pkl_name: str) -> Any:
        print('Loading ' + pkl_name + '...')
        with open('./Cache/' + pkl_name, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save(cls, obj: Any, pkl_name: str) -> None:
        print('Saving ' + pkl_name + '...')
        with open('./Cache/' + pkl_name, 'wb') as f:
            pickle.dump(obj, f)
