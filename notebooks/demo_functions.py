from typing import Optional, List, Union, OrderedDict
import numpy as np
from re import sub as re_sub
from librosa import load
from pathlib import Path
import csv

__all__ = ['load_audio_file_demo', 'read_csv_file_demo', 'clean_sentence_demo']


def load_audio_file_demo(audio_file: str, sr: int, mono: bool,
                         offset: Optional[float] = 0.0,
                         duration: Optional[Union[float, None]] = None)\
        -> np.ndarray:
    """Loads the data of an audio file.

    :param audio_file: The path of the audio file.
    :type audio_file: str
    :param sr: The sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: The audio data.
    :rtype: numpy.ndarray
    """
    return load(path=audio_file, sr=sr, mono=mono,
                offset=offset, duration=duration)[0]


def dump_numpy_object_demo(np_obj: np.ndarray,
                           file_name: Path,
                           ext: Optional[str] = '.npy',
                           replace_ext: Optional[bool] = True) -> None:
    """Dumps a numpy object to HDD.

    :param np_obj: The numpy object.
    :type np_obj: numpy.ndarray
    :param file_name: The file name to be used.
    :type file_name: pathlib.Path
    :param ext: The extension for the dumped object.
    :type ext: str
    :param replace_ext: Replace extension if `file_name`\
                        has a different one?
    :type replace_ext: bool
    """
    f_name = file_name.with_suffix(ext) \
        if replace_ext and (file_name.suffix != ext or file_name.suffix == "") \
        else file_name
    np.save(f'{f_name}', np_obj)


def read_csv_file_demo(file_name: str,
                       base_dir: Optional[Union[str, Path]] = 'csv_files') \
        -> List[OrderedDict]:
    """Reads a CSV file.

    :param file_name: The full file name of the CSV.
    :type file_name: str
    :param base_dir: The root dir of the CSV files.
    :type base_dir: str|pathlib.Path
    :return: The contents of the CSV of the task.
    :rtype: list[collections.OrderedDict]
    """
    file_path = Path().joinpath(base_dir, file_name)
    with file_path.open(mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return [csv_line for csv_line in csv_reader]


def clean_sentence_demo(sentence: str,
                        keep_case: Optional[bool] = False,
                        remove_punctuation: Optional[bool] = True,
                        remove_specials: Optional[bool] = True) -> str:
    """Cleans a sentence.

    :param sentence: Sentence to be clean.
    :type sentence: str
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from sentence?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Cleaned sentence.
    :rtype: str
    """
    the_sentence = sentence if keep_case else sentence.lower()

    # Remove any forgotten space before punctuation and double space.
    the_sentence = re_sub(r'\s([,.!?;:"](?:\s|$))', r'\1', the_sentence).replace('  ', ' ')

    if remove_specials:
        the_sentence = the_sentence.replace('<SOS> ', '').replace('<sos> ', '')
        the_sentence = the_sentence.replace(' <EOS>', '').replace(' <eos>', '')

    if remove_punctuation:
        the_sentence = re_sub('[,.!?;:\"]', '', the_sentence)

    return the_sentence


