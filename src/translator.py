from typing import Optional
from src.hf import load_model_from_huggingface
from langdetect import detect
import tensorflow as tf
import numpy as np
import streamlit as st
from typing import Literal
import requests

# prevent error tensorflow
from src.utils import *


ENGLISH = "en"
INDONESIA = "id"
UNKNOWN = "unknown"
BOTH = "both"

def translate(text: Optional[str], lang: str = "id") -> Optional[str]:
    if text:
        if lang == "id":
            model = load_model_from_huggingface(
                repo_id="putuwaw/text-translation", filename="model.keras"
            )
            eng_vectorizer = load_model_from_huggingface(
                repo_id="putuwaw/text-translation", filename="eng_vectorizer.keras"
            )
            idn_vectorizer = load_model_from_huggingface(
                repo_id="putuwaw/text-translation", filename="idn_vectorizer.keras"
            )

            eng_text_vect_layer = eng_vectorizer.layers[-1]
            eng_vocab = eng_text_vect_layer.get_vocabulary()
            eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
            max_decoded_sentence_length = 20

            # start prediction
            idn_vec = idn_vectorizer.predict(tf.constant([text]))
            eng_start = eng_vectorizer.predict(tf.constant(["[start]"]))

            decoded_sentence = []

            for i in range(max_decoded_sentence_length):
                predictions = model.predict(
                    # Example:
                    # encoder_inputs => saya berangkat ke kampus => [4324, 43, 23, 54]
                    # decoder_inputs => [start] i go to campus => [1, 23, 54, 32, 98]
                    {"encoder_inputs": idn_vec, "decoder_inputs": eng_start},
                    verbose=0,
                )

                predicted_id = np.argmax(predictions[0, i, :])
                predicted_word = eng_index_lookup[predicted_id]

                if predicted_word == "[end]":
                    break

                decoded_sentence.append(predicted_word)
                eng_start = eng_vectorizer(
                    tf.constant([" ".join(["[start]"] + decoded_sentence)])
                )

            return " ".join(decoded_sentence)

        if lang == "en":
            model = load_model_from_huggingface(
                repo_id="putuwaw/text-translation", filename="model_en_ver.keras"
            )
            eng_vectorizer = load_model_from_huggingface(
                repo_id="putuwaw/text-translation",
                filename="eng_vectorizer_en_ver.keras",
            )
            idn_vectorizer = load_model_from_huggingface(
                repo_id="putuwaw/text-translation",
                filename="idn_vectorizer_en_ver.keras",
            )

            idn_text_vect_layer = idn_vectorizer.layers[-1]
            idn_vocab = idn_text_vect_layer.get_vocabulary()
            idn_index_lookup = dict(zip(range(len(idn_vocab)), idn_vocab))
            max_decoded_sentence_length = 20

            # start prediction
            eng_vec = eng_vectorizer.predict(tf.constant([text]))
            idn_start = idn_vectorizer.predict(tf.constant(["[start]"]))

            decoded_sentence = []

            for i in range(max_decoded_sentence_length):
                predictions = model.predict(
                    # Example:
                    # encoder_inputs => i go to campus => [4324, 43, 23, 54]
                    # decoder_inputs => [start] saya berangkat ke kampus => [1, 23, 54, 32, 98]
                    {"encoder_inputs": eng_vec, "decoder_inputs": idn_start},
                    verbose=0,
                )

                predicted_id = np.argmax(predictions[0, i, :])
                predicted_word = idn_index_lookup[predicted_id]

                if predicted_word == "[end]":
                    break

                decoded_sentence.append(predicted_word)
                idn_start = idn_vectorizer(
                    tf.constant([" ".join(["[start]"] + decoded_sentence)])
                )

            return " ".join(decoded_sentence)

    return None


@st.cache_data
def _get_english_words() -> set[str]:
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json"
    response = requests.get(url)
    if response.status_code == 200:
        words_dict = response.json()
        english_words = set(words_dict.keys())
        return english_words
    else:
        return set()


@st.cache_data
def _get_indonesian_words() -> set[str]:
    url = "https://raw.githubusercontent.com/Wikidepia/indonesian_datasets/master/dictionary/wordlist/data/wordlist.txt"
    response = requests.get(url)
    if response.status_code == 200:
        response = response.text
        indonesian_words = set(response.split())
        return indonesian_words
    else:
        return set()


@st.cache_data
def _predict_lang(text: str) -> Literal["both", "en", "id", "unknown"]:
    english_words = _get_english_words()
    indonesian_words = _get_indonesian_words()

    total_idn = 0
    total_eng = 0
    splitted = text.lower().split()
    for word in splitted:
        if word in english_words:
            total_eng += 1
        if word in indonesian_words:
            total_idn += 1

    length_sentence = len(splitted)
    half_sentence = length_sentence // 2

    if total_idn > half_sentence and total_eng > half_sentence:
        return BOTH
    elif total_eng > half_sentence:
        return ENGLISH
    elif total_idn > half_sentence:
        return INDONESIA
    else:
        return UNKNOWN


def verify_lang(text: str, comparison: str) -> bool:
    predicted = _predict_lang(text)
    if predicted == BOTH:
        return comparison
    return predicted
