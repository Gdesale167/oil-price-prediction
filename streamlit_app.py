"""Streamlit dashboard.  Run:  streamlit run streamlit_app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oilpred import database
from oilpred.config import MIN_ROWS, resolve_ticker
from oilpred.data import DataUnavailable, collect, get_prices
from oilpred.explain import mean_abs_shap, shap_values
from oilpred.models import available_models
from oilpred.train import train_model

st.set_page_config(page_title="Oil Price Prediction", page_icon="🛢️", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(ticker: str, refresh_key: int) -> pd.Series:
    return get_prices(ticker, refresh=True)


@st.cache_resource(show_spinner=False)
def fit(ticker: str, model_name: str, test_size: float, last_date: str):
    prices = load_prices(ticker, 0)
    return train_model(prices, model_name, test_size)


st.title("🛢️ Oil Price Prediction")
st.caption("Time-series forecasting with Ridge, XGBoost and LSTM, explained with SHAP.")

with st.sidebar:
    st.header("Settings")
    name = "Brent"
    symbol = resolve_ticker(name)
    st.markdown("**Commodity:** Brent crude oil")
    model_name = st.selectbox("Model", available_models(), index=1)
    days = st.slider("Forecast horizon (business days)", 5, 60, 30)
    test_size = st.slider("Test set (%)", 10, 40, 20) / 100
    if "refresh" not in st.session_state:
        st.session_state.refresh = 0
    if st.button("🔄 Collect latest data"):
        try:
            new = collect(name, "1mo")
            st.session_state.refresh += 1
            load_prices.clear()
            st.success(f"{new} new rows saved to SQLite")
        except Exception as exc:
            st.error(f"Collection failed: {exc}")

try:
    with st.spinner("Loading data..."):
        prices = load_prices(symbol, st.session_state.refresh)
except DataUnavailable as exc:
    st.error(str(exc))
    st.stop()

age_days = (pd.Timestamp.now().normalize() - prices.index[-1]).days
if age_days > 5:
    st.info(
        f"Showing stored data up to {prices.index[-1].date()}. Live refresh from "
        "Yahoo Finance was unavailable, so this is the latest saved snapshot."
    )

if len(prices) < MIN_ROWS:
    st.warning(f"Only {len(prices)} rows stored; need {MIN_ROWS}. Click 'Collect latest data'.")
    st.stop()

with st.spinner(f"Training {model_name}..."):
    model, metrics, test_frame = fit(symbol, model_name, test_size, str(prices.index[-1].date()))

fc = model.forecast(prices, days)
last = float(prices.iloc[-1])
end = float(fc["predicted_price"].iloc[-1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest price", f"${last:,.2f}", help=f"As of {prices.index[-1].date()}")
c2.metric(f"Forecast +{days}d", f"${end:,.2f}", f"{(end / last - 1) * 100:+.2f}%")
c3.metric("Test MAE", f"${metrics['mae']:,.2f}",
          f"{metrics['mae'] - metrics['baseline_mae']:+.2f} vs baseline", delta_color="inverse")
c4.metric("Direction accuracy", f"{metrics['directional_accuracy']:.1f}%")

tab_fc, tab_cmp, tab_shap, tab_db = st.tabs(
    ["📈 Forecast", "🏁 Model comparison", "🔍 Explainability (SHAP)", "🗄️ Database"]
)

with tab_fc:
    fig = go.Figure()
    recent = prices.iloc[-250:]
    fig.add_scatter(x=recent.index, y=recent, name="Historical", line=dict(color="#3b82f6"))
    fig.add_scatter(x=fc.index, y=fc["upper"], line=dict(width=0), showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["lower"], fill="tonexty", name="95% band",
                    line=dict(width=0), fillcolor="rgba(249,115,22,0.18)", hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["predicted_price"], name="Forecast",
                    line=dict(color="#f97316", dash="dash"))
    fig.update_layout(height=450, yaxis_title="USD / barrel", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download forecast CSV", fc.round(4).to_csv().encode(),
                       "oil_forecast.csv", "text/csv")
    st.caption("The band widens with the horizon. Longer forecasts are much less reliable.")

with tab_cmp:
    st.write("Same data, same chronological split, every model:")
    if st.button("Run comparison"):
        rows, curves = [], {}
        for m in available_models():
            _, met, tf = fit(symbol, m, test_size, str(prices.index[-1].date()))
            rows.append({"model": m, **{k: round(v, 3) for k, v in met.items()}})
            curves[m] = tf["predicted"]
            curves["actual"] = tf["actual"]
        st.dataframe(pd.DataFrame(rows).set_index("model"), width="stretch")
        fig = go.Figure()
        for k, s in curves.items():
            fig.add_scatter(x=s.index, y=s, name=k,
                            line=dict(color="black", width=2) if k == "actual" else None)
        fig.update_layout(height=420, yaxis_title="USD / barrel", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.info("`baseline_mae` is the error of 'tomorrow = today'. A model that cannot "
                "beat it adds no value. Oil prices are close to a random walk.")
    st.subheader(f"{model_name}: actual vs predicted (test period)")
    fig = go.Figure()
    fig.add_scatter(x=test_frame.index, y=test_frame["actual"], name="Actual")
    fig.add_scatter(x=test_frame.index, y=test_frame["predicted"], name="Predicted")
    fig.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

with tab_shap:
    if model_name != "xgboost":
        st.info("SHAP explanations are available for the XGBoost model. Select it in the sidebar.")
    else:
        X, values = shap_values(prices, model)
        imp = mean_abs_shap(X, values)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
        fig.update_layout(height=420, title="Mean |SHAP value| (impact on predicted return)")
        st.plotly_chart(fig, width="stretch")

        import matplotlib.pyplot as plt
        import shap

        shap.summary_plot(values, X, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Each dot is a day. Red = high feature value. Right = pushes the "
                   "predicted return up.")

with tab_db:
    st.subheader("Model runs")
    st.dataframe(database.recent_runs(10), width="stretch")
    st.subheader("Stored prices")
    st.dataframe(prices.sort_index(ascending=False).to_frame(), width="stretch")
