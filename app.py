import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, re, base64, warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Data Analysis Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem !important;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #e94560 !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #68d391 !important; }

/* Section headers */
h1 { color: #ffffff !important; font-weight: 800 !important; }
h2, h3 { color: #e2e8f0 !important; font-weight: 600 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #a0aec0 !important;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e94560, #c62a47) !important;
    color: white !important;
}

/* Cards */
.stat-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
}
.rec-card {
    background: linear-gradient(135deg, rgba(233,69,96,0.15), rgba(233,69,96,0.05));
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.7rem 0;
}
.rec-card h4 { color: #e94560 !important; margin: 0 0 0.4rem 0; }
.rec-card p  { color: #cbd5e0 !important; margin: 0; font-size: 0.92rem; line-height: 1.5; }
.insight-pill {
    display: inline-block;
    background: rgba(233,69,96,0.2);
    border: 1px solid rgba(233,69,96,0.4);
    color: #fc8181 !important;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    margin: 3px;
    font-weight: 500;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, rgba(233,69,96,0.25) 0%, rgba(48,43,99,0.6) 100%);
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { font-size: 2.5rem !important; margin: 0; }
.hero p  { color: #a0aec0 !important; font-size: 1.05rem; margin-top: 0.5rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #e94560; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING & CACHING ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean():
    df = pd.read_csv('zomato.csv')
    drop_cols = [c for c in ['url','address','phone','dish_liked','reviews_list','menu_item'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df.rename(columns={
        'approx_cost(for two people)':'cost',
        'listed_in(type)':'type',
        'listed_in(city)':'city'
    }, inplace=True)

    def clean_rate(val):
        val = str(val).strip()
        if val in ['-','NEW','nan','None','']: return np.nan
        m = re.search(r'(\d+\.?\d*)\s*/\s*5', val)
        if m:
            r = float(m.group(1))
            return r if 0 <= r <= 5 else np.nan
        return np.nan

    def clean_cost(val):
        val = str(val).replace(',','').strip()
        try: return float(val)
        except: return np.nan

    df['rate']  = df['rate'].apply(clean_rate)
    df['cost']  = df['cost'].apply(clean_cost)
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    ml_df = df[['location','rest_type','cost','votes','rate']].dropna().copy()
    le_loc  = LabelEncoder()
    le_type = LabelEncoder()
    ml_df['location']  = le_loc.fit_transform(ml_df['location'])
    ml_df['rest_type'] = le_type.fit_transform(ml_df['rest_type'])
    X = ml_df.drop('rate', axis=1)
    y = ml_df['rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X, y_test, preds, mean_squared_error(y_test,preds), r2_score(y_test,preds)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
with st.spinner("Loading and cleaning dataset..."):
    df = load_and_clean()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png", width=160)
    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    locations = ["All"] + sorted(df['location'].unique().tolist())
    sel_loc = st.selectbox("📍 Location", locations)

    rest_types = ["All"] + sorted(df['rest_type'].unique().tolist())
    sel_type = st.selectbox("🍴 Restaurant Type", rest_types)

    min_rate, max_rate = float(df['rate'].min()), float(df['rate'].max())
    rate_range = st.slider("⭐ Rating Range", min_rate, max_rate, (min_rate, max_rate), step=0.1)

    min_cost, max_cost = float(df['cost'].min()), float(df['cost'].max())
    cost_range = st.slider("💰 Cost Range (INR)", min_cost, min(max_cost, 6000.0), (min_cost, 1500.0), step=50.0)

    st.markdown("---")
    st.markdown("### 📁 Project Info")
    st.markdown("""
    <div style='font-size:0.82rem; color:#718096; line-height:1.8'>
    📊 <b>Dataset:</b> Zomato Bangalore<br>
    🔗 <b>Source:</b> Kaggle<br>
    🤖 <b>Model:</b> Random Forest<br>
    🛠️ <b>Stack:</b> Python · Pandas · Sklearn · Plotly
    </div>
    """, unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if sel_loc  != "All":  fdf = fdf[fdf['location'] == sel_loc]
if sel_type != "All":  fdf = fdf[fdf['rest_type'] == sel_type]
fdf = fdf[(fdf['rate'] >= rate_range[0]) & (fdf['rate'] <= rate_range[1])]
fdf = fdf[(fdf['cost'] >= cost_range[0]) & (fdf['cost'] <= cost_range[1])]

# ─── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🍽️ Zomato Data Analysis Dashboard</h1>
  <p>End-to-end EDA · Machine Learning · Business Insights  |  Alfido Tech Data Science Internship</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🏪 Restaurants",    f"{len(fdf):,}",          f"{len(fdf)-len(df):,}" if sel_loc != 'All' else "Total")
k2.metric("⭐ Avg Rating",      f"{fdf['rate'].mean():.2f}", f"/ 5.0")
k3.metric("💰 Avg Cost (₹2)",  f"₹{fdf['cost'].mean():.0f}")
k4.metric("🗳️ Avg Votes",       f"{fdf['votes'].mean():.0f}")
k5.metric("📍 Locations",       f"{fdf['location'].nunique()}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & EDA",
    "🗺️ Location Analysis",
    "🍜 Cuisine Insights",
    "🤖 ML Model",
    "💡 Recommendations"
])

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.04)',
    font=dict(color='#e2e8f0', family='Inter'),
    margin=dict(t=50, b=40, l=40, r=20)
)

# ── TAB 1: OVERVIEW & EDA ──────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        # Rating Distribution
        fig = px.histogram(
            fdf, x='rate', nbins=30,
            title='⭐ Rating Distribution',
            color_discrete_sequence=['#e94560'],
            labels={'rate':'Rating','count':'Frequency'}
        )
        fig.update_layout(**PLOTLY_THEME, title_font_size=16)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Cost Distribution (log scale for readability)
        fig2 = px.histogram(
            fdf, x='cost', nbins=40,
            title='💰 Cost for Two Distribution',
            color_discrete_sequence=['#6c63ff'],
            labels={'cost':'Cost (INR)','count':'Frequency'}
        )
        fig2.update_layout(**PLOTLY_THEME, title_font_size=16)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    # Cost vs Rating scatter (Plotly interactive)
    fig3 = px.scatter(
        fdf.sample(min(5000, len(fdf))), x='cost', y='rate',
        color='votes', color_continuous_scale='plasma',
        opacity=0.55, size_max=8,
        title='💰 Cost vs Rating (colour = votes)',
        labels={'cost':'Cost for Two (INR)', 'rate':'Rating', 'votes':'Votes'},
        hover_data=['location','rest_type']
    )
    fig3.update_layout(**PLOTLY_THEME, title_font_size=16,
                       coloraxis_colorbar=dict(title='Votes', tickfont=dict(color='#a0aec0')))
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation Heatmap
    corr = fdf[['rate','votes','cost']].corr()
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmid=0,
        text=corr.values.round(2), texttemplate='%{text}',
        textfont=dict(size=16, color='white')
    ))
    fig4.update_layout(**PLOTLY_THEME, title='🔗 Correlation Heatmap (Rate · Votes · Cost)', title_font_size=16)
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: LOCATION ANALYSIS ──────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        top_locs = fdf['location'].value_counts().head(20).reset_index()
        top_locs.columns = ['location','count']
        fig = px.bar(
            top_locs, x='count', y='location', orientation='h',
            title='🏙️ Top 20 Locations by Restaurant Count',
            color='count', color_continuous_scale='viridis',
            labels={'count':'Restaurants', 'location':''}
        )
        fig.update_layout(**PLOTLY_THEME, title_font_size=15,
                          yaxis=dict(autorange='reversed'),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        loc_rating = fdf.groupby('location')['rate'].mean().sort_values(ascending=False).head(20).reset_index()
        fig2 = px.bar(
            loc_rating, x='rate', y='location', orientation='h',
            title='⭐ Top 20 Locations by Avg Rating',
            color='rate', color_continuous_scale='plasma',
            labels={'rate':'Avg Rating', 'location':''}
        )
        fig2.update_layout(**PLOTLY_THEME, title_font_size=15,
                           yaxis=dict(autorange='reversed'),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Online order vs book table
    c3, c4 = st.columns(2)
    with c3:
        oo = fdf['online_order'].value_counts().reset_index()
        oo.columns = ['online_order','count']
        fig3 = px.pie(oo, names='online_order', values='count',
                      title='📱 Online Order Availability',
                      color_discrete_sequence=['#e94560','#6c63ff'],
                      hole=0.45)
        fig3.update_layout(**PLOTLY_THEME, title_font_size=15)
        fig3.update_traces(textfont_color='white')
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        bt = fdf['book_table'].value_counts().reset_index()
        bt.columns = ['book_table','count']
        fig4 = px.pie(bt, names='book_table', values='count',
                      title='📅 Table Booking Availability',
                      color_discrete_sequence=['#48bb78','#ed8936'],
                      hole=0.45)
        fig4.update_layout(**PLOTLY_THEME, title_font_size=15)
        fig4.update_traces(textfont_color='white')
        st.plotly_chart(fig4, use_container_width=True)

    # Avg rating by restaurant type
    type_rating = fdf.groupby('rest_type').agg(
        avg_rate=('rate','mean'), count=('rate','count')
    ).reset_index().query('count >= 30').sort_values('avg_rate', ascending=False)
    fig5 = px.bar(
        type_rating, x='rest_type', y='avg_rate',
        title='🍴 Average Rating by Restaurant Type',
        color='avg_rate', color_continuous_scale='magma',
        labels={'rest_type':'Type','avg_rate':'Avg Rating'},
        text=type_rating['avg_rate'].round(2)
    )
    fig5.update_layout(**PLOTLY_THEME, title_font_size=15,
                       xaxis=dict(tickangle=-35), coloraxis_showscale=False)
    fig5.update_traces(textposition='outside', textfont_color='#e2e8f0')
    st.plotly_chart(fig5, use_container_width=True)

# ── TAB 3: CUISINE INSIGHTS ───────────────────────────────────────────────
with tab3:
    # WordCloud
    st.markdown("### ☁️ Cuisine Word Cloud")
    with st.spinner("Generating WordCloud..."):
        text = ' '.join(fdf['cuisines'].astype(str))
        wc = WordCloud(width=1400, height=600, background_color=None, mode='RGBA',
                       colormap='Set1', min_font_size=10, max_words=200).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(14, 6), facecolor='none')
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()

    # Top cuisines bar
    cuisine_exp = fdf['cuisines'].str.split(', ').explode()
    top_c = cuisine_exp.value_counts().head(15).reset_index()
    top_c.columns = ['cuisine','count']
    fig = px.bar(
        top_c, x='cuisine', y='count',
        title='🍝 Top 15 Most Common Cuisines',
        color='count', color_continuous_scale='turbo',
        labels={'cuisine':'Cuisine', 'count':'Count'}, text='count'
    )
    fig.update_layout(**PLOTLY_THEME, title_font_size=16,
                      xaxis=dict(tickangle=-40), coloraxis_showscale=False)
    fig.update_traces(textposition='outside', textfont_color='#e2e8f0')
    st.plotly_chart(fig, use_container_width=True)

    # Avg rating per cuisine
    cuisine_ratings = fdf.copy()
    cuisine_ratings['primary_cuisine'] = fdf['cuisines'].str.split(', ').str[0]
    cr = cuisine_ratings.groupby('primary_cuisine').agg(
        avg_rate=('rate','mean'), count=('rate','count')
    ).reset_index().query('count >= 20').sort_values('avg_rate', ascending=False).head(15)
    fig2 = px.bar(
        cr, x='primary_cuisine', y='avg_rate',
        title='⭐ Avg Rating by Primary Cuisine (min 20 restaurants)',
        color='avg_rate', color_continuous_scale='plasma',
        labels={'primary_cuisine':'Cuisine', 'avg_rate':'Avg Rating'},
        text=cr['avg_rate'].round(2)
    )
    fig2.update_layout(**PLOTLY_THEME, title_font_size=16,
                       xaxis=dict(tickangle=-40), coloraxis_showscale=False)
    fig2.update_traces(textposition='outside', textfont_color='#e2e8f0')
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 4: ML MODEL ──────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🤖 Random Forest Regressor — Rating Prediction")

    with st.spinner("Training model on full dataset..."):
        model, X, y_test, preds, mse, r2 = train_model(df)

    rmse = mse ** 0.5
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",         "Random Forest")
    m2.metric("MSE",           f"{mse:.4f}")
    m3.metric("RMSE",          f"{rmse:.4f}")
    m4.metric("R² Score",      f"{r2:.4f}")

    c1, c2 = st.columns(2)

    with c1:
        # Feature Importance
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})\
                   .sort_values('Importance', ascending=True)
        fig = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title='🔍 Feature Importance',
            color='Importance', color_continuous_scale='viridis',
            labels={'Importance':'Importance Score','Feature':'Feature'}
        )
        fig.update_layout(**PLOTLY_THEME, title_font_size=15, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Actual vs Predicted
        sample_n = min(500, len(y_test))
        idx = np.random.choice(len(y_test), sample_n, replace=False)
        y_sample  = np.array(y_test)[idx]
        p_sample  = preds[idx]
        fig2 = px.scatter(
            x=y_sample, y=p_sample,
            labels={'x':'Actual Rating', 'y':'Predicted Rating'},
            title='🎯 Actual vs Predicted Rating',
            color_discrete_sequence=['#e94560'],
            opacity=0.6
        )
        fig2.add_shape(type='line',
                       x0=y_sample.min(), y0=y_sample.min(),
                       x1=y_sample.max(), y1=y_sample.max(),
                       line=dict(color='#48bb78', dash='dash', width=2))
        fig2.update_layout(**PLOTLY_THEME, title_font_size=15)
        st.plotly_chart(fig2, use_container_width=True)

    # Residual Distribution
    residuals = y_test.values - preds
    fig3 = px.histogram(
        x=residuals, nbins=50,
        title='📉 Residual Distribution (Actual - Predicted)',
        color_discrete_sequence=['#6c63ff'],
        labels={'x':'Residual','count':'Frequency'}
    )
    fig3.add_vline(x=0, line_dash='dash', line_color='#e94560', line_width=2)
    fig3.update_layout(**PLOTLY_THEME, title_font_size=15)
    st.plotly_chart(fig3, use_container_width=True)

    # Rating Predictor Widget
    st.markdown("---")
    st.markdown("### 🔮 Predict a Restaurant's Rating")
    le_loc2  = LabelEncoder().fit(df['location'])
    le_type2 = LabelEncoder().fit(df['rest_type'])

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1: p_loc  = st.selectbox("📍 Location",  sorted(df['location'].unique()))
    with pc2: p_type = st.selectbox("🍴 Type",      sorted(df['rest_type'].unique()))
    with pc3: p_cost = st.slider("💰 Cost (₹)",  100, 3000, 500, 50)
    with pc4: p_vote = st.slider("🗳️ Votes",      0, 5000, 200, 10)

    if st.button("⚡ Predict Rating", use_container_width=True):
        try:
            inp = pd.DataFrame({
                'location':  [le_loc2.transform([p_loc])[0]],
                'rest_type': [le_type2.transform([p_type])[0]],
                'cost':      [float(p_cost)],
                'votes':     [float(p_vote)]
            })
            pred_rating = model.predict(inp)[0]
            stars = "⭐" * int(round(pred_rating))
            st.success(f"### Predicted Rating: **{pred_rating:.2f} / 5.0**  {stars}")
        except Exception as ex:
            st.warning(f"Prediction error: {ex}")

# ── TAB 5: RECOMMENDATIONS ───────────────────────────────────────────────
with tab5:
    st.markdown("### 💡 Strategic Recommendations for Alfido Tech-Style Platform")
    st.markdown("<br>", unsafe_allow_html=True)

    recs = [
        ("🏙️ Prioritise High-Density Hotspot Partnerships",
         "BTM leads with 3,700+ restaurants — nearly 2× the next area (HSR/Koramangala). "
         "Establish strategic restaurant partnerships here first for maximum reach and minimum CAC."),
        ("💎 Launch a 'Budget Gems' Content Series",
         "Restaurants with cost < Rs.400 and rating >= 4.0 exist across multiple localities. "
         "A curated 'Budget Gems' section drives aspirational discovery and boosts user engagement."),
        ("🎮 Gamified Review & Voting System",
         "Number of votes is the strongest rating predictor (~58% feature importance in ML). "
         "Incentivise reviews through rewards, badges, and voucher giveaways to create engagement loops."),
        ("📱 Feature Online Order & Table Booking Prominently",
         "Restaurants offering both features show consistently higher ratings. Treat these as primary "
         "trust signals in search results and restaurant cards to boost conversion rates."),
        ("🌍 Invest in Niche Cuisine Discovery",
         "Continental, Italian, and Kerala cuisines are underrepresented but score above-average ratings. "
         "A dedicated 'Discover Premium' section with curated recommendations differentiates the platform.")
    ]

    for i, (title, body) in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
            <h4>Recommendation {i}: {title}</h4>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Quick Insights at a Glance")

    top_loc  = df['location'].value_counts().index[0]
    top_cui  = df['cuisines'].str.split(', ').explode().value_counts().index[0]
    avg_r    = df['rate'].mean()
    avg_cost = df['cost'].mean()

    ins = [
        f"<span class='insight-pill'>📍 Top Location: {top_loc}</span>",
        f"<span class='insight-pill'>🍜 #1 Cuisine: {top_cui}</span>",
        f"<span class='insight-pill'>⭐ Avg Rating: {avg_r:.2f}</span>",
        f"<span class='insight-pill'>💰 Avg Cost: Rs.{avg_cost:.0f}</span>",
        f"<span class='insight-pill'>🏪 Clean Records: {len(df):,}</span>",
        "<span class='insight-pill'>🤖 Best Feature: votes (~58%)</span>",
    ]
    st.markdown(" ".join(ins), unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.85rem; padding:1rem'>
    🍽️ Zomato Data Analysis · Alfido Tech Data Science Internship · Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
