import time
import random
from io import StringIO

from src.translator import translate, verify_lang
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
lang_dict = {
    "Indonesian": "id",
    "English": "en",
}

warning_placeholder = st.empty()
string_data = None
uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "md"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

col_origin, col_translate = st.columns(2)
with col_origin:
    lang_origin = st.selectbox(
        "Text",
        list(lang_dict),
    )
    if uploaded_file and string_data:
        txt = st.text_area(
            "Text", value=string_data, height=200, label_visibility="collapsed"
        )
    else:
        txt = st.text_area("Text", value=None, height=200, label_visibility="collapsed")
    target_languages = [lang for lang in list(lang_dict) if lang != lang_origin]

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

    if txt:
        if not verify_lang(txt, lang_dict[lang_origin]):
            warning_placeholder.warning(
                f"Please input {lang_origin} language!", icon=":material/warning:"
            )
            st.stop()
        else:
            warning_placeholder.empty()

    translated_text = translate(txt, lang_dict[lang_origin])
    if translated_text:
        translated_display = ""
        for idx, word in enumerate(translated_text.split()):
            translated_display += word + " "
            placeholder.text_area(
                "Translate",
                value=translated_display.strip() + TEXT_CARET,
                **text_area_kwargs,
            )
            if idx < 10:
                time.sleep(random.uniform(0, 0.3))

        st.session_state["translation_result"] = translated_display.strip()

        placeholder.text_area(
            "Translate", value=translated_display.strip(), **text_area_kwargs
        )

        st_copy_to_clipboard(st.session_state["translation_result"])
