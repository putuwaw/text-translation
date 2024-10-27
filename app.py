import streamlit as st

home_page = st.Page("src/home.py", title="Home Page", icon=":material/home:")
model_page = st.Page("src/model.py", title="Model Page", icon=":material/model_training:")
about_page = st.Page("src/about.py", title="About Page", icon=":material/info:")

pg = st.navigation([home_page, model_page, about_page])
pg.run()
