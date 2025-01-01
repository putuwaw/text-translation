from typing import Optional
from src.hf import load_model_from_huggingface
from langdetect import detect
import tensorflow as tf
import numpy as np

# prevent error tensorflow
from src.utils import *


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
                    {"encoder_inputs": idn_vec, "decoder_inputs": eng_start}, verbose=0
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
                repo_id="putuwaw/text-translation", filename="eng_vectorizer_en_ver.keras"
            )
            idn_vectorizer = load_model_from_huggingface(
                repo_id="putuwaw/text-translation", filename="idn_vectorizer_en_ver.keras"
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
                    {"encoder_inputs": eng_vec, "decoder_inputs": idn_start}, verbose=0
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


def verify_lang(text: str, comparison: str) -> bool:
    return detect(text) == comparison
