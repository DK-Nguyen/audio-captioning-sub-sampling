from typing import Optional, List, Union, \
    OrderedDict, MutableSequence, Dict
import numpy as np
from re import sub as re_sub
from librosa import load
from pathlib import Path
import csv
from functools import partial
from collections import Counter
from itertools import chain
import pickle


# ---------- Functions for processing captions ----------
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


def get_sentence_words_demo(sentence: str,
                            unique: Optional[bool] = False,
                            keep_case: Optional[bool] = False,
                            remove_punctuation: Optional[bool] = True,
                            remove_specials: Optional[bool] = True) -> List[str]:
    """Splits input sentence into words.

    :param sentence: Sentence to split
    :type sentence: str
    :param unique: Returns a list of unique words.
    :type unique: bool
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from sentence?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Sentence words
    :rtype: list[str]
    """
    words = clean_sentence_demo(
        sentence, keep_case=keep_case,
        remove_punctuation=remove_punctuation,
        remove_specials=remove_specials).strip().split()

    if unique:
        words = list(set(words))

    return words


def get_words_counter_demo(captions: MutableSequence[str],
                           use_unique: Optional[bool] = False,
                           keep_case: Optional[bool] = False,
                           remove_punctuation: Optional[bool] = True,
                           remove_specials: Optional[bool] = True) -> Counter:
    """Creates a Counter object from the\
    words in the captions.

    :param captions: The captions.
    :type captions: list[str]|iterable
    :param use_unique: Use unique only words from the captions?
    :type use_unique: bool
    :param keep_case: Keep capitals and small (True) or turn\
                      everything to small case (False)
    :type keep_case: bool
    :param remove_punctuation: Remove punctuation from captions?
    :type remove_punctuation: bool
    :param remove_specials: Remove special tokens?
    :type remove_specials: bool
    :return: Counter object from\
             the words in the captions.
    :rtype: collections.Counter
    """
    partial_func = partial(
        get_sentence_words_demo,
        unique=use_unique, keep_case=keep_case,
        remove_punctuation=remove_punctuation,
        remove_specials=remove_specials
    )
    return Counter(chain.from_iterable(map(partial_func, captions)))


def dump_pickle_file_demo(obj: object, file_name: Union[str, Path],
                          protocol: Optional[int] = 2) -> None:
    """Dumps an object to pickle file.

    :param obj: The object to dump.
    :type obj: object | list | dict | numpy.ndarray
    :param file_name: The resulting file name.
    :type file_name: str|pathlib.Path
    :param protocol: The protocol to be used.
    :type protocol: int
    """
    str_file_name = file_name if type(file_name) == str else str(file_name)

    with open(str_file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def load_pickle_file_demo(file_name: Path,
                          encoding='latin1')\
        -> Union[object, List, Dict, np.ndarray]:
    """Loads a pickle file.

    :param file_name: File name (extension included).
    :type file_name: pathlib.Path
    :param encoding: Encoding of the file.
    :type encoding: str
    :return: Loaded object.
    :rtype: object | list | dict | numpy.ndarray
    """
    with file_name.open('rb') as f:
        return pickle.load(f, encoding=encoding)


# ---------- Functions for processing audio files ----------
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