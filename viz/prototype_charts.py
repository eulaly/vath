'''
    prototype_charts.py
    generate test charts for later addition to streamlit
'''
   

from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np

inp = Path(r"c:/users/moses/projects/vath/analysis/jobs/f452-1/review.csv")
out = Path("viz/")
out.mkdir(parents=True, exist_ok=True)

stance_order = ["support", "oppose", "neutral", "unknown"]

# tone_order = ["positive", "negative", "neutral", "mixed", "unknown", "unclear"]
# default order was actually better - unclear/negative/neutral/mixed/positive vs unknown/oppose/neutral/support
# same for pct w/in stance
df = pd.read_csv(inp)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["date_day"] = df["date"].dt.date
df["stance"] = df["stance"].fillna("unknown")
df["tone"] = df["tone"].fillna("unknown")

# 1. stance share
counts = df["stance"].value_counts().reindex(stance_order, fill_value=0).reset_index()
counts.columns = ["stance", "count"]
fig = px.bar(counts, x="count", y="stance", orientation="h", text="count")
fig.write_html(out / "stance_share.html")

# 2. stance over time
daily = df.groupby(["date_day", "stance"]).size().reset_index(name="count")
fig = px.bar(daily, x="date_day", y="count", color="stance", category_orders={"stance": stance_order})
fig.write_html(out / "stance_over_time.html")

# 3. stance x tone
heat = df.groupby(["stance", "tone"]).size().reset_index(name="count")
fig = px.density_heatmap(heat, x="tone", y="stance", z="count", category_orders={"stance": stance_order})
fig.write_html(out / "stance_tone_heatmap.html")

# 4. confidence by stance
fig = px.box(df, x="stance", y="stance_confidence", category_orders={"stance": stance_order}, points="outliers")
fig.write_html(out / "confidence_by_stance.html")

# 5. cumulative stance and share over time
daily = (
    df.groupby(["date_day", "stance"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=stance_order, fill_value=0)
      .sort_index()
)

cum = daily.cumsum()
cum_long = cum.reset_index().melt(id_vars="date_day", var_name="stance", value_name="cumulative_count")

fig = px.area(
    cum_long,
    x="date_day",
    y="cumulative_count",
    color="stance",
    category_orders={"stance": stance_order},
    title="cumulative comments by stance over time",
)
fig.write_html(out / "cumulative_stance_area.html")

cum_pct = cum.div(cum.sum(axis=1), axis=0).reset_index().melt(
    id_vars="date_day", var_name="stance", value_name="cumulative_share"
)

fig = px.line(
    cum_pct,
    x="date_day",
    y="cumulative_share",
    color="stance",
    category_orders={"stance": stance_order},
    title="cumulative stance share over time",
)
fig.update_yaxes(tickformat=".0%")
fig.write_html(out / "cumulative_stance_share.html")

# 7. diverging h-bar
stance_counts = df["stance"].value_counts().reindex(stance_order, fill_value=0)

div = pd.DataFrame({
    "stance": ["oppose", "support", "neutral", "unknown"],
    "count": [
        -stance_counts.get("oppose", 0),
         stance_counts.get("support", 0),
         stance_counts.get("neutral", 0),
         stance_counts.get("unknown", 0),
    ],
})

fig = px.bar(
    div,
    x="count",
    y="stance",
    orientation="h",
    text=div["count"].abs(),
    title="support vs oppose",
)
fig.update_xaxes(title="comments", zeroline=True)
fig.update_traces(textposition="outside")
fig.write_html(out / "stance_diverging_bar.html")

# 8. Stance x Tone labels
heat = pd.crosstab(df["stance"], df["tone"]).reindex(
    index=stance_order,
    columns=[c for c in tone_order if c in df["tone"].unique()],
    fill_value=0,
)

fig = px.imshow(
    heat,
    text_auto=True,
    aspect="auto",
    title="stance x tone, count",
)
fig.write_html(out / "stance_tone_counts.html")

rowpct = heat.div(heat.sum(axis=1).replace(0, np.nan), axis=0)

fig = px.imshow(
    rowpct,
    text_auto=".0%",
    aspect="auto",
    title="stance x tone, percent within stance",
)
fig.write_html(out / "stance_tone_rowpct.html")


