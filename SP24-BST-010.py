import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ==========================================
# 1. PREMIUM NEON UI & ANIMATION ENGINE
# ==========================================
st.set_page_config(page_title="DataPulse: Advanced Statistical Analytics v11.0", page_icon="🔮", layout="wide")

st.markdown(
    """
<style>
/* Return to Original Deep Purple Radial Gradient */
.stApp { 
    background: radial-gradient(circle at 20% 30%, #1a0b2e 0%, #070312 100%); 
    color: #e0d1ff; 
}

/* Sidebar Styling with Original Blur */
[data-testid="stSidebar"] { 
    background: rgba(26, 11, 46, 0.4) !important; 
    backdrop-filter: blur(20px); 
}

/* Original Sidebar Navigation Hover Effects */
[data-testid="stSidebarNav"] li {
    transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55);
    border-radius: 15px; margin: 10px; background: rgba(124, 58, 237, 0.05);
}

[data-testid="stSidebarNav"] li:hover {
    transform: scale(1.1) translateZ(30px);
    box-shadow: 0 0 20px #7c3aed, inset 0 0 10px #f472b6;
}

/* Original Neon Stat Cards */
.stat-card {
    background: rgba(255, 255, 255, 0.03); 
    backdrop-filter: blur(15px);
    border: 1px solid rgba(167, 139, 250, 0.3); 
    border-radius: 25px;
    padding: 30px; 
    text-align: center; 
    transition: 0.5s all ease;
}

.stat-card:hover { 
    transform: translateY(-12px); 
    border-color: #f472b6; 
    box-shadow: 0 15px 40px rgba(124, 58, 237, 0.5); 
}

/* Profile Picture Neon Glow */
.profile-pic {
    width: 200px; height: 200px;
    border-radius: 50%;
    border: 4px solid #7c3aed;
    box-shadow: 0 0 20px #7c3aed, 0 0 40px #f472b6;
    object-fit: cover;
    display: block; margin-left: auto; margin-right: auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. FAIL-SAFE CORE
# ==========================================
if "df_master" not in st.session_state:
    st.session_state.df_master = None


def ZenithGuard():
    if st.session_state.df_master is None:
        st.markdown(
            '<div style="background:rgba(239, 68, 68, 0.1); border: 2px solid #ef4444; padding: 60px; border-radius: 25px; text-align: center; font-size: 1.8rem; font-weight: 900;">⚠️ PLEASE INGEST DATASTREAM IN SIDEBAR TO ACTIVATE</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    return st.session_state.df_master


# ==========================================
# 3. SIDEBAR WORKBENCH
# ==========================================
with st.sidebar:
    st.markdown(
        "<h1 style='color:#a78bfa; text-align:center;'>🔮 DATAPULSE PRO</h1>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("📥 UPLOAD DATA", type=["csv", "xlsx"])
    if up:
        st.session_state.df_master = (
            pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        )
        st.success("SYNAPSE LINKED")

    menu = st.radio(
        "NAVIGATE",
        [
            "🏠 Home",
            "⚙️ Transformation",
            "📊 Correlations",
            "🧠 Classification",
            "📈 Regression",
            "🔮 Forecast",
            "🧪 Hypothesis",
            "👤 My CV",
            "📖 Guide",
        ],
    )

# ==========================================
# 4. MODULE: HOME & DATA SANITATION
# ==========================================
# ==========================================
# 4. MODULE: HOME & DATA SANITATION
# ==========================================
# ==========================================
# 4. MODULE: HOME & DATA SANITATION
# ==========================================
if menu == "🏠 Home":
    st.title("🏠 Dashboard Home")
    if st.session_state.df_master is not None:
        df = st.session_state.df_master
        total_nulls = df.isnull().sum().sum()
        integrity = (1 - total_nulls / df.size) * 100

        st.markdown(
            f'<div style="background:linear-gradient(90deg, rgba(124,58,237,0.2), transparent); padding:25px; border-left:8px solid #7c3aed; border-radius:15px; margin-bottom:30px; font-size:1.3rem;">🧬 Pulse Summary: Processing <b>{len(df)}</b> observations. Integrity: <b>{integrity:.1f}%</b>.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🛠️ DATA SANITATION ENGINE"):
            st.warning(f"Detected {total_nulls} missing synapses in the datastream.")
            action = st.radio(
                "Repair Strategy", ["None", "Drop Nulls", "Fill with Median (Numeric)"]
            )
            if st.button("Apply Repair"):
                if action == "Drop Nulls":
                    st.session_state.df_master = df.dropna()
                elif action == "Fill with Median (Numeric)":
                    num_cols = df.select_dtypes(include=np.number).columns
                    st.session_state.df_master[num_cols] = df[num_cols].fillna(
                        df[num_cols].median()
                    )
                st.rerun()

            # --- NEW GRAPH ADDED HERE ---
            st.markdown("---")
            if st.checkbox("Show Null-Value Map"):
                st.markdown("### ❄️ Data Completeness Map")
                
                # Beautification: Detailed Description
                st.info("This heatmap provides a visual audit of dataset integrity. "
                        "The highlighted lines indicate 'synapse gaps' (missing values) "
                        "that may require repair using the strategies above.")
                
                fig_null, ax_null = plt.subplots(figsize=(10, 2))
                
                # Dark Theme Calibration: Blending with your #070312 background
                fig_null.patch.set_facecolor("#070312")
                ax_null.set_facecolor("#070312")
                
                # Heatmap styling with Magma for high contrast
                sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='magma', ax=ax_null)
                
                # Styling text to match your UI
                plt.xticks(color="#e0d1ff")
                st.pyplot(fig_null)
                st.caption("Visualization: Highlighted lines represent missing data points.")
            # -----------------------------

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="stat-card"><h3>Rows</h3><h1>{len(df)}</h1></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><h3>Cols</h3><h1>{len(df.columns)}</h1></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="stat-card"><h3>Nulls</h3><h1>{total_nulls}</h1></div>',
                unsafe_allow_html=True,
            )
        st.dataframe(df.head(50), use_container_width=True)
    else:
        ZenithGuard()

# ==========================================
# 5. MODULE: TRANSFORMATION
# ==========================================
elif menu == "⚙️ Transformation":
    st.title("⚙️ Neural Type Transformation")
    df = ZenithGuard()
    col_to_change = st.selectbox("Select Column to Transform", df.columns)
    current_type = df[col_to_change].dtype
    st.write(f"**Current State:** `{current_type}`")
    target_type = st.radio(
        "Target Neural Type",
        ["String (Object)", "Integer", "Float", "DateTime", "Category"],
    )

    if st.button("⚡ Execute Transformation"):
        try:
            if target_type == "String (Object)":
                st.session_state.df_master[col_to_change] = df[col_to_change].astype(
                    str
                )
            elif target_type == "Integer":
                st.session_state.df_master[col_to_change] = (
                    pd.to_numeric(df[col_to_change], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
            elif target_type == "Float":
                st.session_state.df_master[col_to_change] = pd.to_numeric(
                    df[col_to_change], errors="coerce"
                ).astype(float)
            elif target_type == "DateTime":
                st.session_state.df_master[col_to_change] = pd.to_datetime(
                    df[col_to_change], errors="coerce"
                )
            elif target_type == "Category":
                st.session_state.df_master[col_to_change] = df[col_to_change].astype(
                    "category"
                )
            st.success("Successfully recalibrated type.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")


# ==========================================
# 6. MODULE: CORRELATIONS
# ==========================================
elif menu == "📊 Correlations":
    st.title("📊 Statistical Insights")
    df = ZenithGuard()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(num_cols) >= 2:
        # Creating tabs to organize the 5 different visualizations
        tab1, tab2 = st.tabs(["Global Heatmap", "Distribution & Density"])
        
        with tab1:
            st.markdown("### 🌡️ Association Matrix")
            method = st.selectbox("Correlation Algorithm", ["pearson", "kendall", "spearman"])
            corr = df[num_cols].corr(method=method)
            
            # Interactive Heatmap using Plotly
            fig = px.imshow(
                corr, 
                text_auto=True, 
                color_continuous_scale='Magma', 
                template="plotly_dark",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Currently utilizing **{method.capitalize()}** coefficients to map linear and non-linear associations.")

        with tab2:
            st.markdown("### 🧬 Univariate & Multivariate Analysis")
            col_to_plot = st.selectbox("Select Primary Variable for Analysis", num_cols)
            
            # Row 1: Boxen and Violin Plots
            c1, c2 = st.columns(2)
            with c1:
                fig1, ax1 = plt.subplots()
                fig1.patch.set_facecolor("#070312")
                ax1.set_facecolor("#070312")
                sns.boxenplot(y=df[col_to_plot], color="#f472b6", ax=ax1)
                
                # Dark Theme Styling
                ax1.tick_params(colors='#e0d1ff')
                for spine in ax1.spines.values():
                    spine.set_color('#7c3aed')
                
                st.pyplot(fig1)
                st.caption("Boxen Plot: Advanced quantile distribution analysis for outlier detection.")

            with c2:
                fig2, ax2 = plt.subplots()
                fig2.patch.set_facecolor("#070312")
                ax2.set_facecolor("#070312")
                sns.violinplot(y=df[col_to_plot], color="#7c3aed", ax=ax2)
                
                # Dark Theme Styling
                ax2.tick_params(colors='#e0d1ff')
                for spine in ax2.spines.values():
                    spine.set_color('#f472b6')
                
                st.pyplot(fig2)
                st.caption("Violin Plot: Kernel density estimation showing data concentration peaks.")

            # Row 2: Joint Density and CDF Plots
            st.markdown("---")
            st.markdown("### 📊 Probability & Density Mapping")
            c3, c4 = st.columns(2)
            
            with c3:
                # Joint Density Hexbin Plot
                feat_x = st.selectbox("X-Axis (Density)", num_cols, index=0, key="hx_new")
                feat_y = st.selectbox("Y-Axis (Density)", num_cols, index=min(1, len(num_cols)-1), key="hy_new")
                
                fig_joint, ax_joint = plt.subplots()
                fig_joint.patch.set_facecolor("#070312")
                ax_joint.set_facecolor("#070312")
                
                hb = ax_joint.hexbin(df[feat_x], df[feat_y], gridsize=15, cmap='magma', mincnt=1)
                ax_joint.tick_params(colors='#e0d1ff')
                ax_joint.set_xlabel(feat_x, color='#e0d1ff')
                ax_joint.set_ylabel(feat_y, color='#e0d1ff')
                
                st.pyplot(fig_joint)
                st.caption("Hexbin Plot: Multivariate density mapping showing data clusters.")

            with c4:
                # Cumulative Distribution Function (CDF)
                fig_cdf, ax_cdf = plt.subplots()
                fig_cdf.patch.set_facecolor("#070312")
                ax_cdf.set_facecolor("#070312")
                
                sns.ecdfplot(data=df[col_to_plot], color="#f472b6", ax=ax_cdf)
                ax_cdf.tick_params(colors='#e0d1ff')
                ax_cdf.set_xlabel(col_to_plot, color='#e0d1ff')
                ax_cdf.set_ylabel("Accumulated Probability", color='#e0d1ff')
                
                for spine in ax_cdf.spines.values():
                    spine.set_color('#7c3aed')
                
                st.pyplot(fig_cdf)
                st.caption("CDF: Statistical probability that the variable is ≤ a specific value.")
    else:
        st.warning("Analysis Inhibited: Datastream requires at least 2 numeric columns.")
# ==========================================
# 7. MODULE: CLASSIFICATION
# ==========================================
elif menu == "🧠 Classification":
    st.title("🧠 Logic Pipeline")
    df = ZenithGuard()
    cat_cols, num_cols = (
        df.select_dtypes(exclude=np.number).columns.tolist(),
        df.select_dtypes(include=np.number).columns.tolist(),
    )
    if cat_cols and num_cols:
        y_col, x_cols = st.selectbox("Target", cat_cols), st.multiselect(
            "Predictors", num_cols
        )
        if x_cols and st.button("🚀 Train"):
            le = LabelEncoder()
            y = le.fit_transform(df[y_col].dropna())
            X = df[x_cols].loc[df[y_col].dropna().index]
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression())]
            ).fit(X, y)
            st.markdown(
                f'<div class="stat-card"><h3>Accuracy: {accuracy_score(y, pipe.predict(X)):.4f}</h3></div>',
                unsafe_allow_html=True,
            )
            cm = confusion_matrix(y, pipe.predict(X))
            st.plotly_chart(
                px.imshow(
                    cm,
                    text_auto=True,
                    x=le.classes_,
                    y=le.classes_,
                    template="plotly_dark",
                )
            )

# ==========================================
# 8. MODULE: REGRESSION
# ==========================================
elif menu == "📈 Regression":
    st.title("📈 Regression Engine")
    df = ZenithGuard()
    num = df.select_dtypes(include=np.number).columns
    y_var, x_vars = st.selectbox("Target (Y)", num), st.multiselect(
        "Predictors (X)", num
    )
    # Add this inside 'elif menu == "📈 Regression":' before the 'Train Model' button
    with st.expander("🔍 EXPLORE RELATIONSHIPS"):
        st.markdown("### Feature Interaction Matrix")
    fig_scat = px.scatter_matrix(df, dimensions=x_vars, color=y_var, 
                                 template="plotly_dark", opacity=0.6)
    fig_scat.update_layout(height=600)
    st.plotly_chart(fig_scat, use_container_width=True)
    if x_vars and st.button("Train Model"):
        model = LinearRegression().fit(df[x_vars], df[y_var])
        y_pred = model.predict(df[x_vars])
        st.markdown(
            f'<div class="stat-card"><h3>R² Score: {r2_score(df[y_var], y_pred):.4f}</h3></div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#070312")
        ax.set_facecolor("#070312")
        plt.scatter(y_pred, df[y_var] - y_pred, color="#a78bfa", alpha=0.6)
        plt.axhline(y=0, color="#f472b6", linestyle="--")
        st.pyplot(fig)
        st.caption("Residual Plot: Points should be randomly scattered.")

# ==========================================
# 9. MODULE: FORECAST
# ==========================================
elif menu == "🔮 Forecast":
    st.title("🔮 Projection")
    df = ZenithGuard()
    num_list = df.select_dtypes(include=np.number).columns
    sel = st.selectbox("Timeline Metric", num_list)
    if df[sel].nunique() > 1:
        try:
            model = ExponentialSmoothing(df[sel].dropna(), trend="add").fit()
            fc = model.forecast(30)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(y=df[sel].values, name="Actual", line=dict(color="#7c3aed"))
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(df), len(df) + 30),
                    y=fc,
                    name="Forecast",
                    line=dict(color="#f472b6", dash="dot"),
                )
            )
            st.plotly_chart(
                fig.update_layout(template="plotly_dark", title="30-Step Projection")
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 10. MODULE: HYPOTHESIS
# ==========================================
elif menu == "🧪 Hypothesis":
    st.title("🧪 Inferential Workbench")
    df = ZenithGuard()
    col = st.selectbox("Variable", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#070312")
    ax.set_facecolor("#070312")
    sns.histplot(df[col].dropna(), kde=True, color="#f472b6", ax=ax)
    st.pyplot(fig)
    val = st.number_input("Null Mean", value=float(df[col].mean()))
    t, p = stats.ttest_1samp(df[col].dropna(), val)
    st.markdown(
        f'<div class="stat-card"><h3>P-Value: {p:.5f}</h3><p>{"Significant" if p < 0.05 else "Not Significant"}</p></div>',
        unsafe_allow_html=True,
    )

# ==========================================
# 11. MODULE: CV
# ==========================================
elif menu == "👤 My CV":
    st.title("👤 Professional Profile")
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("profile.jpg", width=200)
        except:
            st.warning("Place 'profile.jpg' in folder.")
        st.markdown(
            f'<div class="stat-card"><h2>Hajira Shahbaz</h2><p>BS Statistics (Data Science)</p><hr>📧 SP24-BST-010@cuilahore.edu.pk<br>📞 0321-8490566</div>',
            unsafe_allow_html=True,
        )
        st.write("● 🏏 Cricket | ● 📷 Photography")
    with col2:
        st.markdown(
            """### 🎓 COMSATS University Lahore\n* **BS Statistics** (Data Science Specialist)\n### 🎓 Beaconhouse High School\n* **O/A Levels**\n### 💼 DataPulse Pro\n* Developed high-performance Neural Engine suite."""
        )

# ==========================================
# 12. MODULE: SUMMARY & DOCUMENTATION
# ==========================================
elif menu == "📖 Guide": 
    st.title("📋 System Architecture & Technical Summary")
    
    # Hero Section - Detailed Project Summary visible on the page
    st.markdown(
        f"""
        <div class="stat-card" style="text-align:left;">
        <h2>Project Overview: DataPulse Intelligence Suite</h2>
        <p>DataPulse is a comprehensive analytical ecosystem developed by <b>Hajira Shahbaz</b> for the <b>BS Statistics (Data Science)</b> program at <b>COMSATS University Lahore</b>. 
        The system bridges the gap between raw data ingestion and high-level inferential modeling, providing 
        a professional-grade interface for complex Data Science workflows.</p>
        
        <hr style="border: 0.5px solid rgba(167, 139, 250, 0.2); margin: 20px 0;">
        
        <h4>Executive Summary</h4>
        <p>The DataPulse dashboard represents a sophisticated integration of statistical theory and modern software engineering. 
        Designed to automate the standard Data Science pipeline—<b>Ingest → Clean → Visualize → Model → Forecast</b>—it serves as a 
        robust tool for academic and professional data auditing. The "Cyber-Dark" aesthetic (Radial Gradient #1a0b2e to #070312) 
        is specifically calibrated for professional durability and modern appeal.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Technical Deep Dive Columns
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🛠️ Data Governance & Preprocessing")
        st.info(
            "**1. Failsafe Ingestion:** Utilizes the 'ZenithGuard' core to prevent execution errors when no data is present.\\\n"
            "**2. Neural Transformation:** Allows for real-time recalibration of data types (Categorical, Numerical, Temporal) to ensure model compatibility.\\\n"
            "**3. Sanitation Logic:** Employs median imputation to handle missing synapses without skewing the distribution mean."
        )
        
        st.markdown("### 📊 Advanced Statistical Analysis")
        st.success(
            "**Multivariate Correlations:** Imploys Pearson (linear), Spearman (rank), and Kendall (ordinal) coefficients to map associations.\\\n"
            "**Distributional Rigor:** Uses Boxen plots for quantile-level outlier detection and CDF plots for probability accumulation.\\\n"
            "**Density Analysis:** Features Joint Density Hexbin plots to visualize data concentration peaks in high-volume datasets."
        )

    with col_b:
        st.markdown("### 🧠 Predictive Modeling & ML")
        st.warning(
            "**Regression Engine:** Implements Ordinary Least Squares (OLS) with interactive residual diagnostics to verify Gaussian assumptions.\\\n"
            "**Logic Pipeline:** A classification framework using Standard Scaling and Logistic Regression for target predictions.\\\n"
            "**Forecasting:** Uses Exponential Smoothing (ETS) to project 30-step future trends based on additive seasonality."
        )
        
        st.markdown("### 🎨 System UX Design")
        st.write(
            "**UI/UX:** Built on a 'Cyber-Dark' radial gradient palette (#070312) to reduce visual fatigue during prolonged analysis.\\\n"
            "**Interactivity:** Leverages Streamlit session state and Plotly's WebGL engine for responsive, low-latency visualization.\\\n"
            "**Beautification:** Custom CSS glassmorphism and neon animation engine for premium user engagement."
        )

    # Statistical Conclusion
    st.markdown("---")
    st.markdown("### 🔬 Final System Conclusion")
    st.write(
        "By integrating both Descriptive and Inferential statistics, DataPulse v11.0 provides a rigorous framework for data-driven decision making. "
        "It successfully demonstrates the application of advanced statistical visualizations and machine learning within a modern web-based interface."
    )
    st.caption("Developed by Hajira Shahbaz | DataPulse Analytics Suite | COMSATS University Lahore | © 2026")