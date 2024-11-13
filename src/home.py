import time
import random
from src.translator import translate
import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("Text Translation")

if "translation_result" not in st.session_state:
    st.session_state["translation_result"] = None

TEXT_CARET = "\u258c"
lang_list = ["Indonesia", "English"]

col_origin, col_translate = st.columns(2)
with col_origin:
    lang_origin = st.selectbox(
        "Text",
        lang_list,
    )
    txt = st.text_area("Text", value=None, height=200, label_visibility="collapsed")
    target_languages = [lang for lang in lang_list if lang != lang_origin]

with col_translate:
    lang_translate = st.selectbox(
        "Translate",
        target_languages,
    )

    text_area_kwargs = {
        "height": 200,
        "disabled": True,
        "label_visibility": "collapsed",
    }

    placeholder = st.empty()
    placeholder.text_area("Translate", value=None, **text_area_kwargs)

    translated_text = translate(txt)
    if translated_text:
        translated_display = ""
        for word in translated_text.split():
            translated_display += word + " "
            placeholder.text_area(
                "Translate",
                value=translated_display.strip() + TEXT_CARET,
                **text_area_kwargs
            )
            time.sleep(random.uniform(0, 0.3))

        st.session_state["translation_result"] = translated_display.strip()

        placeholder.text_area(
            "Translate", value=translated_display.strip(), **text_area_kwargs
        )

        st_copy_to_clipboard(st.session_state["translation_result"])
