# streamlit run analysis/viz/prototype_streamlit.py
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

df = pd.read_csv(r"analysis/jobs/f452-1/review.csv")
st.set_page_config(layout="wide")

stance = st.multiselect("Filter stance", sorted(df["stance"].dropna().unique()), default=sorted(df["stance"].dropna().unique()))
q = st.text_input("Search comment text")
dff = df[df["stance"].isin(stance)]
if q:
    dff = dff[dff["text"].fillna("").str.contains(q, case=False, regex=False)]

st.dataframe(dff[["comment_id", "title", "stance", "stance_confidence", "tone"]], width="stretch")
st.write("Showing " + str(len(dff))+ " comments")

cid = st.selectbox("comment", dff["comment_id"].astype(str))
row = dff[dff["comment_id"].astype(str) == cid].iloc[0]

st.subheader(row["title"])
st.write(row["text"])
st.write(row["author"] + ", " + row["date"][:10])
st.write("**model:** " + str(row["model"]))
st.markdown("**stance:** " + str(row["stance"]) + "  \n**confidence:** " + str(row["stance_confidence"]) + "  \n**tone:** " + str(row["tone"]))
st.write("**analysis:** "+ row["stance_rationale"])
