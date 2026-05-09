# streamlit run viz/streamlit.py -- --jobs-dir analysis/jobs/f452-1
import argparse
from pathlib import Path
from datetime import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--jobs-dir", default="analysis/jobs/f452-1", type=Path,
                    help="Job directory containing review.csv, forum.jsonl, and prompt.txt")
args, _ = parser.parse_known_args()  # parse_known_args: ignore Streamlit's own argv entries
workdir = args.jobs_dir
df = pd.read_csv(workdir/"review.csv")
df['date_dt'] = pd.to_datetime(df.date)
df["date_day"] = df["date_dt"].dt.date
forum = pd.read_json(workdir/"forum.jsonl", lines=True).iloc[0].to_dict()
prompt = (workdir/"prompt.txt").read_text(encoding="utf-8")

stance_colors = {'oppose':'#ffa15a', 'neutral':'#e377c2','support':'#19d3f3','unknown':'#000000'}
stance_order = ["oppose", "mixed", "unknown", "neutral", "support"]

st.set_page_config(layout="wide")
st.title("Virginia Townhall Explorer",anchor=None)
st.caption("Explore data collected from Virginia's public comment system. Source code at https://github.com/eulaly/vath")

st.subheader("Proposal",anchor=None,divider="gray")
st.markdown(f"**{forum.get('reg_title')}**")
st.text(forum.get('reg_desc'))
st.caption(f'Comments posted from {dt.strftime(min(df.date_dt),"%D")}—{dt.strftime(max(df.date_dt),"%D")} at https://www.townhall.virginia.gov/L/Comments.cfm?GDocForumID={forum.get("forum_id")}')

st.subheader("Comment Summary",anchor=False,divider="gray")
summary_left, summary_right = st.columns([1,2])
with summary_left:
# Summary Table
    summary_stats = (
    df.groupby("stance").size()
      .reindex(stance_order, fill_value=0)
      .reset_index(name="count")
      .assign(percent=lambda d: (d["count"] / d["count"].sum()).map("{:.1%}".format))
)

    st.dataframe(summary_stats, hide_index=True, width="stretch")
with summary_right:
# Stance div-h
    counts = df["stance"].value_counts()
    stance_divh = go.Figure()
    stance_divh.add_bar(y=["stance"], x=[-counts.get("oppose",0)], name="oppose", orientation="h", marker_color=stance_colors.get('oppose'), text=[counts.get("oppose",0)], textposition="inside")
    stance_divh.add_bar(y=["stance"], x=[counts.get("neutral",0)], name="neutral", orientation="h", marker_color=stance_colors.get('neutral'), text=[counts.get("neutral",0)], textposition="inside")
    stance_divh.add_bar(y=["stance"], x=[counts.get("unknown",0)], name="unknown", orientation="h", marker_color=stance_colors.get('unknown'), text=[counts.get("unknown",0)], textposition="inside")
    stance_divh.add_bar(y=["stance"], x=[counts.get("support",0)], name="support", orientation="h", marker_color=stance_colors.get('support'), text=[counts.get("support",0)], textposition="inside")
    stance_divh.update_yaxes(title_text="",showticklabels=False)
    stance_divh.update_layout(barmode="relative", title="", height=180, margin=dict(l=0,r=0,t=0,b=0),xaxis_title="", yaxis_title="",legend=dict(orientation="v",y=0.12))
    st.plotly_chart(stance_divh,width='stretch')

# Daily Comments Breakdown, 3 Tabs
daily_wide = (
    df.groupby(["date_day", "stance"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=stance_order, fill_value=0)
      .sort_index()
)

daily_long = (
    daily_wide.reset_index()
      .melt(id_vars="date_day", var_name="stance", value_name="count")
)

cum_wide = daily_wide.cumsum()

cum_long = (
    cum_wide.reset_index()
      .melt(id_vars="date_day", var_name="stance", value_name="cumulative_count")
)

cum_total = cum_wide.sum(axis=1)
cum_share = cum_wide.div(cum_total.where(cum_total > 0), axis=0)

cum_share_long = (
    cum_share.reset_index()
      .melt(id_vars="date_day", var_name="stance", value_name="cumulative_share")
)


tab_daily, tab_area, tab_share = st.tabs([
    "Daily",
    "Cumulative",
    "Cumulative Share",
])

with tab_daily:
    fig = px.bar(
        daily_long,
        x="date_day",
        y="count",
        color="stance",
        category_orders={"stance": stance_order},
        color_discrete_map=stance_colors,
    )
    fig.update_layout(barmode="stack", height=420, legend_orientation="v")
    st.plotly_chart(fig, width="stretch")

with tab_area:
    fig = px.area(
        cum_long,
        x="date_day",
        y="cumulative_count",
        color="stance",
        category_orders={"stance": stance_order},
        color_discrete_map=stance_colors,
    )
    fig.update_layout(height=420, legend_orientation="v")
    st.plotly_chart(fig, width="stretch")

with tab_share:
    fig = px.line(
        cum_share_long,
        x="date_day",
        y="cumulative_share",
        color="stance",
        category_orders={"stance": stance_order},
        color_discrete_map=stance_colors,
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=420, legend_orientation="v")
    st.plotly_chart(fig, width="stretch")
    
st.subheader("Comment Explorer",anchor=False,divider="gray") 
# comment explorer
cex_left, cex_right = st.columns([1,1])
with cex_left:
    filter_stance = st.multiselect("Filter stance", sorted(df["stance"].dropna().unique()), default=sorted(df["stance"].dropna().unique()))
    filter_tone = st.multiselect("Filter tone", sorted(df["tone"].dropna().unique()), default=sorted(df["tone"].dropna().unique()))
    dff = df[df["stance"].isin(filter_stance) & df["tone"].isin(filter_tone)]

with cex_right:
    q = st.text_input("Search comment title and text")
    if q:
        dff = dff[dff["text"].fillna("").str.contains(q, case=False, regex=False)]
    st.text(""); st.text("")
    st.text("Showing " + str(len(dff))+ " comments",text_alignment="right", width="stretch")

st.dataframe(dff[["comment_id", "title", "text", "stance", "stance_confidence", "tone"]], width="stretch")

cid = st.selectbox("Select comment to view:", dff["comment_id"].astype(str))
row = dff[dff["comment_id"].astype(str) == cid].iloc[0]

st.markdown(f'**{row["title"]}**')
st.text(row["text"])
st.write(row["author"] + ", " + row["date_dt"].strftime("%D"))

st.divider()

st.subheader('Analysis')
cexs_left, cexs_right = st.columns([1,1])
with cexs_left:
    st.write(f"**stance:** {row['stance']}")
    st.write(f"**stance_confidence:** {row['stance_confidence']:.2f}")
    st.write(f"**tone:** {row['tone']}")
    st.write("**analysis:** "+ row["stance_rationale"])
with cexs_right:
    x_order = ["unknown","oppose","mixed","neutral","support"]  # includes mixed even if absent; harmless zero column
    y_order = ["positive","neutral","mixed","negative","unclear"]
    tab = pd.crosstab(df["tone"], df["stance"]).reindex(index=y_order, columns=x_order, fill_value=0)
    pct = tab.div(tab.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
    tone_stance = px.imshow(
        pct,
        x=x_order, y=y_order,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Greens",
    )
    tone_stance.update_traces(text=tab.astype(str) + " / " + (pct*100).round(0).astype(int).astype(str) + "%")
    tone_stance.add_scatter(x=[row["stance"]],y=[row["tone"]],mode="markers",marker=dict(size=15,color="yellow",symbol="cross",line=dict(width=1, color="red")),showlegend=False)
    tone_stance.update_layout(height=420, xaxis_title="stance", yaxis_title="tone")
    st.plotly_chart(tone_stance, width='stretch')
    st.caption("Tone by stance, % within tone", text_alignment="right",width="stretch")

st.divider()
st.write("**model:** " + str(row["model"]))
with st.expander("Prompt", expanded=False):
    st.code(prompt, language="text")

tone_conf = px.box(df,x="stance",y="stance_confidence",color="stance",category_orders={"stance":stance_order},color_discrete_map=stance_colors,points="outliers",title="Comment Stance Classification Confidence")
tone_conf.update_yaxes(range=[0,1.02])
tone_conf.update_layout(height=430, legend_orientation="v")
st.plotly_chart(tone_conf,width="stretch")
