import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

import streamlit as st

st.header("Text Translation")
st.write('''Text translation is a web app that can translate between Indonesian and English language using Transformer.
            This web app was created to fulfill the final semester assignment of ML for Text subject.''')

st.subheader("Tech Stack")
st.write("The tech stack used in this project are:")
tech1, tech2, tech3, tech4 = st.columns(4)
with tech1:
    st.image("https://github.com/streamlit.png", width=100)
    st.write("[Streamlit](https://streamlit.io/)")
with tech2:
    st.image("https://github.com/keras-team.png", width=100)
    st.write("[Keras](https://keras.io/)")
with tech3:
    st.image("https://github.com/huggingface.png", width=100)
    st.write("[Hugging Face](https://huggingface.co)")
with tech4:
    st.image("https://github.com/tensorflow.png", width=100)
    st.write("[Tensorflow](https://www.tensorflow.org/)")

st.subheader("Contributors")
person1, person2, person3 = st.columns(3)
with person1:
    st.image("https://github.com/putuwaw.png", width=100)
    st.write("[Putu Widyantara](https://github.com/putuwaw)")
with person2:
    st.image("https://github.com/AksidF.png", width=100)
    st.write("[Diska Fortunawan](https://github.com/AksidF)")
with person3:
    st.image("https://github.com/OdeArdika.png", width=100)
    st.write("[Ode Ardika](https://github.com/OdeArdika)")


st.subheader("Source Code")
st.write(
    "The source code can be found [here](https://github.com/putuwaw/text-translation).")