"""
AutoML Data Analyzer Dashboard
A production-level Streamlit application for automated data analysis,
visualization, and machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /*
   * Fonts: Playfair Display (elegant serif for headings) +
   *        Nunito (rounded, warm, highly legible for body)
   * Background: Deep warm charcoal — #111110 base, not cold blue-black
   */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&family=Nunito:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    font-weight: 400;
    letter-spacing: 0.01em;
  }

  /* ── Main background — warm charcoal, not cold navy ── */
  .stApp { background: #111110; color: #e5e2dc; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #181816 !important;
    border-right: 1px solid #262520;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Nunito', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5a574f;
    margin-top: 1.5rem;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #1c1b18 0%, #111110 55%, #191815 100%);
    border: 1px solid #2a2823;
    border-radius: 18px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -90px; right: -70px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(212,175,100,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 38%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,180,141,0.06) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    line-height: 1.15;
    background: linear-gradient(135deg, #e8d5a3 0%, #c9a84c 45%, #a8893a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.45rem 0;
  }
  .hero-sub {
    color: #756f63;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.6;
    margin: 0;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(212,175,100,0.1);
    border: 1px solid rgba(212,175,100,0.22);
    color: #c9a84c;
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Section cards ── */
  .section-card {
    background: #181816;
    border: 1px solid #262520;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
  }
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #d4c9b0;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #1c1b18 !important;
    border: 1px solid #2a2823 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
  }
  [data-testid="stMetricLabel"] {
    color: #756f63 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
  }
  [data-testid="stMetricValue"] {
    color: #c9a84c !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Upload box ── */
  [data-testid="stFileUploader"] {
    background: #181816 !important;
    border: 2px dashed #2a2823 !important;
    border-radius: 14px !important;
  }

  /* ── Selectbox & inputs ── */
  .stSelectbox > div > div {
    background: #1c1b18 !important;
    border-color: #2a2823 !important;
    color: #e5e2dc !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #b8922e, #9a7826) !important;
    color: #fdf8ee !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.5rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #c9a84c, #b8922e) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 22px rgba(185,146,46,0.30) !important;
  }

  /* ── Divider ── */
  hr { border-color: #262520 !important; margin: 1.5rem 0 !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #181816 !important;
    border: 1px solid #262520 !important;
    border-radius: 10px !important;
  }

  /* ── Alerts ── */
  .stSuccess { background: rgba(99,180,141,0.1) !important; border-color: rgba(99,180,141,0.3) !important; }
  .stInfo    { background: rgba(212,175,100,0.08) !important; border-color: rgba(212,175,100,0.25) !important; }
  .stWarning { background: rgba(220,130,60,0.10) !important; border-color: rgba(220,130,60,0.28) !important; }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div > div { background: #c9a84c !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: rgba(99,180,141,0.12) !important;
    color: #63b48d !important;
    border: 1px solid rgba(99,180,141,0.28) !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
  }

  /* ── Tag pill ── */
  .tag-pill {
    display: inline-block;
    background: rgba(212,175,100,0.1);
    border: 1px solid rgba(212,175,100,0.22);
    color: #c9a84c;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 2px;
    letter-spacing: 0.04em;
  }
  .tag-green  { background: rgba(99,180,141,0.1); border-color: rgba(99,180,141,0.25); color: #63b48d; }
  .tag-yellow { background: rgba(220,130,60,0.10); border-color: rgba(220,130,60,0.25); color: #dc823c; }

  /* ── Stat row ── */
  .stat-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 0.5rem 0;
  }
  .stat-item {
    background: #1c1b18;
    border: 1px solid #2a2823;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: #756f63;
    min-width: 120px;
  }
  .stat-item strong {
    display: block;
    color: #d4c9b0;
    font-size: 1.1rem;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
  }

  /* ── Plotly chart background ── */
  .js-plotly-plot .plotly .main-svg { background: transparent !important; }

  /* ── Headings inside markdown ── */
  h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
    color: #d4c9b0 !important;
    letter-spacing: -0.01em !important;
  }
  h3 { font-size: 1.25rem !important; }
  h4 { font-family: 'Nunito', sans-serif !important; font-weight: 700 !important; color: #a8a090 !important; letter-spacing: 0.02em !important; }

  /* ── Sidebar wordmark ── */
  .sidebar-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 800;
    font-style: italic;
    background: linear-gradient(135deg, #e8d5a3, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
</style>
""", unsafe_allow_html=True)

# ─── Sklearn imports (lazy; prevents crash if not installed) ─────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error,
    mean_squared_error,
)

# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8a8070", family="Nunito, sans-serif"),
    xaxis=dict(gridcolor="#262520", zerolinecolor="#262520"),
    yaxis=dict(gridcolor="#262520", zerolinecolor="#262520"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# Warm gold + sage green + terracotta palette
COLOR_SEQ = ["#c9a84c", "#63b48d", "#dc823c", "#a07850",
             "#8fba9f", "#e8c87a", "#b07048", "#5a9e7a"]


def detect_problem_type(series: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    if series.dtype == "object" or series.nunique() <= 15:
        return "classification"
    return "regression"


def preprocess(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Full preprocessing pipeline. Returns X_train, X_test, y_train, y_test, feature_names."""
    X = df.drop(columns=[target])
    y = df[target].copy()

    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")

    # Separate numeric and categorical columns
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Impute
    num_imp = SimpleImputer(strategy="median")
    if num_cols:
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    if cat_cols:
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        # Encode categorical
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if classification
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target)

    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names


def get_classifiers():
    return {
        "🔵 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "🌳 Decision Tree": DecisionTreeClassifier(random_state=42),
        "🔴 SVM": SVC(probability=True, random_state=42),
    }


def get_regressors():
    return {
        "📈 Linear Regression": LinearRegression(),
        "🌲 Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "🌳 Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    }


def eval_classifier(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "CV Score (mean)": cv.mean(),
        "CV Score (std)": cv.std(),
        "y_pred": y_pred,
        "model": model,
    }


def eval_regressor(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    mse = mean_squared_error(y_test, y_pred)
    return {
        "R² Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "CV Score (mean)": cv.mean(),
        "CV Score (std)": cv.std(),
        "y_pred": y_pred,
        "model": model,
    }


def feature_importance_chart(model, feature_names: list, top_n: int = 15):
    """Return a Plotly figure for feature importance (tree models)."""
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig = go.Figure(go.Bar(
        x=imp[idx][::-1],
        y=[feature_names[i] for i in idx][::-1],
        orientation="h",
        marker=dict(
            color=imp[idx][::-1],
            colorscale=[[0, "#2a2823"], [1, "#c9a84c"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        **PLOT_THEME,
        height=max(300, top_n * 26),
        xaxis_title="Importance",
        yaxis_title="",
    )
    return fig


def confusion_matrix_chart(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test) | set(y_pred))
    fig = px.imshow(
        cm,
        text_auto=True,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        color_continuous_scale=[[0, "#111110"], [0.5, "#6b4a1a"], [1, "#c9a84c"]],
        title="Confusion Matrix",
    )
    fig.update_layout(**PLOT_THEME, coloraxis_showscale=False)
    return fig


def correlation_heatmap(df_num: pd.DataFrame):
    corr = df_num.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, "#1c1b18"], [0.5, "#7a5a20"], [1, "#c9a84c"]],
        title="Correlation Heatmap",
        zmin=-1, zmax=1,
    )
    fig.update_layout(**PLOT_THEME, height=max(350, len(corr) * 40))
    return fig


def target_dist_chart(series: pd.Series, problem_type: str):
    if problem_type == "classification":
        vc = series.value_counts().reset_index()
        vc.columns = ["label", "count"]
        fig = px.bar(
            vc, x="label", y="count",
            color="count",
            color_continuous_scale=[[0, "#2a2823"], [1, "#c9a84c"]],
            title="Target Distribution",
        )
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False, xaxis_title="Class", yaxis_title="Count")
    else:
        fig = px.histogram(
            series, nbins=40,
            title="Target Distribution",
            color_discrete_sequence=["#c9a84c"],
        )
        fig.update_layout(**PLOT_THEME, xaxis_title=series.name, yaxis_title="Frequency")
    return fig


def feature_dist_grid(df: pd.DataFrame, cols: list, max_cols: int = 4):
    """Multi-panel distribution plots for numeric features."""
    cols = cols[:16]  # cap
    n = len(cols)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
    for i, col in enumerate(cols):
        r = i // ncols + 1
        c = i % ncols + 1
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                name=col,
                marker_color="#c9a84c",
                opacity=0.8,
                showlegend=False,
            ),
            row=r, col=c,
        )
    fig.update_layout(
        title="Feature Distributions",
        **PLOT_THEME,
        height=260 * nrows,
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = "#8a8070"
    return fig


def results_csv(metrics: dict, model_name: str, problem_type: str) -> bytes:
    rows = [{"Model": model_name, "Problem": problem_type, "Metric": k, "Value": v}
            for k, v in metrics.items() if k not in ("y_pred", "model")]
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem'>
      <span style='font-family:"Playfair Display",Georgia,serif; font-size:1.35rem;
                   font-weight:800; font-style:italic;
                   background: linear-gradient(135deg,#e8d5a3,#c9a84c);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   background-clip:text;'>
        ⚡ AutoML
      </span>
      <p style='color:#5a574f; font-size:0.68rem; margin:3px 0 0;
                letter-spacing:0.14em; font-family:Nunito,sans-serif;
                font-weight:700; text-transform:uppercase;'>
        Data Analyzer
      </p>
    </div>
    <hr style='border-color:#262520; margin:0.8rem 0'>
    """, unsafe_allow_html=True)

    st.markdown("## Navigation")
    page = st.radio(
        "",
        ["🏠  Overview", "📊  Visualizations", "⚡  Model Training", "📋  Results"],
        label_visibility="collapsed",
    )

    st.markdown("## Settings")
    test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05,
                          help="Fraction of data for testing")
    top_n_feat = st.slider("Top Features to Show", 5, 20, 10)

    st.markdown("---")
    st.markdown("""
    <div style='color:#5a574f; font-size:0.72rem; text-align:center; line-height:1.6;
                font-family:Nunito,sans-serif;'>
      AutoML Analyzer v1.0<br>
      Built with Streamlit + Sklearn
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">⚡ &nbsp;AutoML Platform</div>
  <div class="hero-title">AutoML Data Analyzer</div>
  <p class="hero-sub">Upload any dataset → automatic preprocessing → model training → insights. No code required.</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  FILE UPLOAD
# ════════════════════════════════════════════════════════════════

st.markdown("### 📂 Upload Your Dataset")
uploaded_file = st.file_uploader(
    "Drop a CSV or Excel file here",
    type=["csv", "xlsx", "xls"],
    help="Supports CSV and Excel formats up to ~200 MB",
)

if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#4b5563;'>
      <div style='font-size:3rem; margin-bottom:1rem'>📥</div>
      <div style='font-family:"Playfair Display",Georgia,serif; font-size:1.15rem; color:#756f63; font-style:italic;'>
        Upload a dataset to get started
      </div>
      <p style='font-size:0.85rem; margin-top:0.5rem; color:#5a574f;'>
        Supports CSV and Excel &nbsp;•&nbsp; Auto-detects problem type &nbsp;•&nbsp; No setup required
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Load dataframe ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

with st.spinner("Loading dataset…"):
    df = load_data(uploaded_file)

st.success(f"✅ **{uploaded_file.name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════

if "🏠" in page:
    st.markdown("---")
    st.markdown("### 📋 Dataset Overview")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗂 Rows", f"{df.shape[0]:,}")
    c2.metric("🔢 Columns", df.shape[1])
    c3.metric("🔴 Missing", f"{df.isnull().sum().sum():,}")
    c4.metric("📊 Numeric Cols", df.select_dtypes(include=np.number).shape[1])

    st.markdown("<br>", unsafe_allow_html=True)

    # Preview
    with st.expander("👁️ Preview First Rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🔍 Column Summary")
        summary = pd.DataFrame({
            "Type": df.dtypes.astype(str),
            "Non-Null": df.notnull().sum(),
            "Null %": (df.isnull().mean() * 100).round(1).astype(str) + "%",
            "Unique": df.nunique(),
        })
        st.dataframe(summary, use_container_width=True)

    with col_r:
        st.markdown("#### 📐 Descriptive Stats")
        st.dataframe(df.describe().round(3), use_container_width=True)

    # Missing values bar
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.markdown("#### ⚠️ Missing Values by Column")
        fig = px.bar(
            x=missing.index, y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            color=missing.values,
            color_continuous_scale=[[0, "#2a2823"], [1, "#dc823c"]],
            title="Missing Value Distribution",
        )
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No missing values found in this dataset!")


# ════════════════════════════════════════════════════════════════
#  TARGET COLUMN SELECTION (shared across pages)
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Target Column")
target_col = st.sidebar.selectbox(
    "Select your target variable",
    df.columns.tolist(),
    index=len(df.columns) - 1,
)
problem_type = detect_problem_type(df[target_col])

badge_cls = "tag-green" if problem_type == "classification" else "tag-yellow"
badge_text = "🏷️ Classification" if problem_type == "classification" else "📈 Regression"
st.sidebar.markdown(f'<span class="tag-pill {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ════════════════════════════════════════════════════════════════

if "📊" in page:
    st.markdown("---")
    st.markdown("### 📊 Exploratory Visualizations")

    # Target distribution
    st.markdown("#### 🎯 Target Variable Distribution")
    st.plotly_chart(target_dist_chart(df[target_col], problem_type), use_container_width=True)

    # Correlation heatmap
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        st.markdown("#### 🔗 Correlation Heatmap")
        st.plotly_chart(correlation_heatmap(num_df), use_container_width=True)

    # Feature distributions
    num_cols_list = [c for c in num_df.columns if c != target_col]
    if num_cols_list:
        st.markdown("#### 📦 Feature Distributions")
        st.plotly_chart(feature_dist_grid(df, num_cols_list[:16]), use_container_width=True)

    # Scatter: top 2 numeric features vs target
    if problem_type == "regression" and len(num_cols_list) >= 1:
        st.markdown("#### 🔵 Scatter: Feature vs Target")
        feat_choice = st.selectbox("Pick a feature to plot against target", num_cols_list)
        fig_sc = px.scatter(
            df, x=feat_choice, y=target_col,
            trendline="ols",
            color_discrete_sequence=["#c9a84c"],
            title=f"{feat_choice} vs {target_col}",
        )
        fig_sc.update_layout(**PLOT_THEME)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Box plots
    if problem_type == "classification" and len(num_cols_list) >= 1:
        st.markdown("#### 📦 Feature Box Plots by Class")
        feat_box = st.selectbox("Pick a numeric feature", num_cols_list)
        fig_box = px.box(
            df, x=target_col, y=feat_box,
            color=target_col,
            color_discrete_sequence=COLOR_SEQ,
            title=f"{feat_box} by {target_col}",
        )
        fig_box.update_layout(**PLOT_THEME, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════════════

if "⚡" in page:
    st.markdown("---")
    st.markdown(f"### ⚡ Model Training  —  {badge_text}")

    model_options = get_classifiers() if problem_type == "classification" else get_regressors()
    model_name = st.selectbox("🧠 Choose a model", list(model_options.keys()))

    col_ts, col_btn = st.columns([3, 1])
    with col_ts:
        st.info(f"**Train/Test split:** {int((1-test_size)*100)}% / {int(test_size*100)}%  |  "
                f"**5-Fold Cross-Validation** enabled  |  **Target:** `{target_col}`")

    train_btn = col_btn.button("🚀 Train Model", use_container_width=True)

    if train_btn:
        prog = st.progress(0, text="Preprocessing…")
        try:
            with st.spinner("Preprocessing data…"):
                X_train, X_test, y_train, y_test, feat_names = preprocess(
                    df, target_col, test_size
                )
            prog.progress(30, text="Training model…")

            with st.spinner("Training & evaluating…"):
                chosen_model = model_options[model_name]
                if problem_type == "classification":
                    metrics = eval_classifier(chosen_model, X_train, X_test, y_train, y_test)
                else:
                    metrics = eval_regressor(chosen_model, X_train, X_test, y_train, y_test)
            prog.progress(80, text="Generating visuals…")

            # Store in session state
            st.session_state["metrics"] = metrics
            st.session_state["model_name"] = model_name
            st.session_state["problem_type"] = problem_type
            st.session_state["feat_names"] = feat_names
            st.session_state["y_test"] = y_test
            prog.progress(100, text="Done!")

            # ── Metrics display ──────────────────────────────────────
            st.markdown("#### 📈 Evaluation Metrics")
            metric_keys = [k for k in metrics if k not in ("y_pred", "model", "CV Score (std)")]
            cols = st.columns(len(metric_keys))
            for i, k in enumerate(metric_keys):
                val = metrics[k]
                if isinstance(val, float):
                    cols[i].metric(k, f"{val:.4f}")
                else:
                    cols[i].metric(k, str(val))

            st.markdown(
                f"**Cross-Val:** {metrics['CV Score (mean)']:.4f} ± {metrics['CV Score (std)']:.4f}  "
                f"(5 folds)"
            )

            # ── Confusion matrix ─────────────────────────────────────
            if problem_type == "classification":
                st.markdown("#### 🟪 Confusion Matrix")
                st.plotly_chart(
                    confusion_matrix_chart(y_test, metrics["y_pred"]),
                    use_container_width=True,
                )

            # ── Predictions scatter (regression) ─────────────────────
            if problem_type == "regression":
                st.markdown("#### 🔵 Actual vs Predicted")
                pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": metrics["y_pred"]})
                fig_pv = px.scatter(
                    pred_df, x="Actual", y="Predicted",
                    color_discrete_sequence=["#c9a84c"],
                    title="Actual vs Predicted Values",
                    opacity=0.75,
                )
                fig_pv.add_shape(
                    type="line",
                    x0=pred_df["Actual"].min(), x1=pred_df["Actual"].max(),
                    y0=pred_df["Actual"].min(), y1=pred_df["Actual"].max(),
                    line=dict(color="#63b48d", dash="dash", width=2),
                )
                fig_pv.update_layout(**PLOT_THEME)
                st.plotly_chart(fig_pv, use_container_width=True)

            # ── Feature importance ────────────────────────────────────
            fi_fig = feature_importance_chart(metrics["model"], feat_names, top_n_feat)
            if fi_fig:
                st.markdown("#### 🌟 Feature Importance")
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info("ℹ️ Feature importance is available for tree-based models.")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            prog.empty()


# ════════════════════════════════════════════════════════════════
#  PAGE: RESULTS
# ════════════════════════════════════════════════════════════════

if "📋" in page:
    st.markdown("---")
    st.markdown("### 📋 Results & Insights")

    if "metrics" not in st.session_state:
        st.warning("⚠️ No results yet — train a model first in the **🤖 Model Training** tab.")
        st.stop()

    metrics = st.session_state["metrics"]
    model_name = st.session_state["model_name"]
    ptype = st.session_state["problem_type"]
    feat_names = st.session_state["feat_names"]
    y_test = st.session_state["y_test"]

    # Summary card
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">🏆 Model Summary</div>
      <div class="stat-row">
        <div class="stat-item"><strong>{model_name.split(' ',1)[-1]}</strong>Model</div>
        <div class="stat-item"><strong>{ptype.title()}</strong>Problem Type</div>
        <div class="stat-item"><strong>{len(feat_names)}</strong>Features Used</div>
        <div class="stat-item"><strong>{len(y_test)}</strong>Test Samples</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # All metrics
    st.markdown("#### 📊 Full Metrics Breakdown")
    metric_rows = []
    for k, v in metrics.items():
        if k in ("y_pred", "model"):
            continue
        if isinstance(v, float):
            metric_rows.append({"Metric": k, "Value": round(v, 6)})
    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(metric_df, use_container_width=True)

    # Dataset insights
    st.markdown("#### 💡 Dataset Insights")
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    insight_lines = [
        f"• Dataset has **{df.shape[0]:,} samples** and **{df.shape[1]} features**.",
        f"• **{num_df.shape[1]} numeric** and **{cat_df.shape[1]} categorical** columns.",
        f"• Missing values: **{df.isnull().sum().sum():,}** total.",
    ]

    if ptype == "classification":
        n_cls = df[target_col].nunique()
        insight_lines.append(f"• Target has **{n_cls} classes** — "
                              f"{'binary' if n_cls == 2 else 'multi-class'} classification.")
        balance = df[target_col].value_counts(normalize=True).std()
        if balance > 0.15:
            insight_lines.append("• ⚠️ Class imbalance detected — consider resampling techniques.")
        else:
            insight_lines.append("• ✅ Classes appear relatively balanced.")
    else:
        skew = df[target_col].skew()
        insight_lines.append(f"• Target skewness: **{skew:.2f}** — "
                              f"{'consider log-transform' if abs(skew) > 1 else 'fairly normal'}.")
        if "R² Score" in metrics:
            r2 = metrics["R² Score"]
            quality = "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "fair" if r2 > 0.5 else "poor"
            insight_lines.append(f"• Model R² = **{r2:.4f}** ({quality} fit).")

    if hasattr(metrics["model"], "feature_importances_"):
        imp = metrics["model"].feature_importances_
        top_feat = feat_names[np.argmax(imp)]
        insight_lines.append(f"• 🌟 Most important feature: **{top_feat}** ({imp.max()*100:.1f}% importance).")

    for line in insight_lines:
        st.markdown(line)

    # Download section
    st.markdown("---")
    st.markdown("#### ⬇️ Download Results")
    col_d1, col_d2 = st.columns(2)

    csv_bytes = results_csv(metrics, model_name, ptype)
    col_d1.download_button(
        label="📥 Download Metrics CSV",
        data=csv_bytes,
        file_name=f"automl_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": metrics["y_pred"]})
    col_d2.download_button(
        label="📥 Download Predictions CSV",
        data=pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
