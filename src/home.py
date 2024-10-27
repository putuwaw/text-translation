import streamlit as st
from src.translator import translate

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Text Translation")

col_origin, col_translate = st.columns(2)

with col_origin:
    lang_origin = st.selectbox(
        "Text",
        ("Indonesia"),
    )
    txt = st.text_area("Text", value=None, height=200, label_visibility="collapsed")

with col_translate:
    lang_translate = st.selectbox(
        "Translate",
        ("English"),
        disabled=True,
    )

    translated_text = translate(txt)
    txt_translate = st.text_area(
        "Translate",
        value=translated_text,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )
