"""
🎯 Smart Listing AI Dashboard
Streamlit-based visualization for Amazon Listing Analysis

Features:
- View analysis results from Google Sheets
- Interactive charts with Plotly
- Benchmarking comparison
- ASIN input for new analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime

# ============================================
# 📋 PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Smart Listing AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🎨 CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Score badges */
    .score-high { color: #00C851; font-weight: bold; }
    .score-medium { color: #ffbb33; font-weight: bold; }
    .score-low { color: #ff4444; font-weight: bold; }
    
    /* ASIN link */
    .asin-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# 🔌 GOOGLE SHEETS CONNECTION
# ============================================
SPREADSHEET_ID = "1_0WrdwdWthtaMHSAiNy8HqpAsTW9xNStTw7o9JDEWWU"

@st.cache_resource
def get_google_credentials():
    """Get Google credentials from Streamlit secrets"""
    try:
        creds_dict = st.secrets["google_credentials"]
        scopes = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return creds
    except Exception as e:
        st.error(f"❌ Помилка авторизації: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sheet_data(sheet_name: str) -> pd.DataFrame:
    """Load data from Google Sheets"""
    try:
        creds = get_google_credentials()
        if not creds:
            return pd.DataFrame()
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        data = worksheet.get_all_records()
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Помилка завантаження {sheet_name}: {e}")
        return pd.DataFrame()

def save_to_config(key: str, value: str) -> bool:
    """Save value to Config sheet"""
    try:
        creds = get_google_credentials()
        if not creds:
            return False
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("Config")
        
        # Find the row with this key
        all_data = worksheet.get_all_values()
        for i, row in enumerate(all_data):
            if row and row[0].strip() == key:
                # Update existing row (column B = index 2 in API)
                worksheet.update_cell(i + 1, 2, value)
                return True
        
        # If key not found, append new row
        worksheet.append_row([key, value, ""])
        return True
        
    except Exception as e:
        st.error(f"❌ Помилка збереження: {e}")
        return False

def load_config_fresh() -> dict:
    """Load configuration from Config sheet (no cache)"""
    try:
        creds = get_google_credentials()
        if not creds:
            return {}
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("Config")
        
        data = worksheet.get_all_records()
        config = {}
        for row in data:
            key = str(row.get("Key", "")).strip()
            value = str(row.get("Value", "")).strip()
            if key:
                config[key] = value
        
        return config
    except Exception as e:
        st.error(f"❌ Помилка завантаження Config: {e}")
        return {}

@st.cache_data(ttl=300)
def load_config() -> dict:
    """Load configuration from Config sheet"""
    try:
        creds = get_google_credentials()
        if not creds:
            return {}
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("Config")
        
        data = worksheet.get_all_records()
        config = {}
        for row in data:
            key = str(row.get("Key", "")).strip()
            value = str(row.get("Value", "")).strip()
            if key:
                config[key] = value
        
        return config
    except Exception as e:
        st.error(f"❌ Помилка завантаження Config: {e}")
        return {}

# ============================================
# 📊 HELPER FUNCTIONS
# ============================================
def parse_score(score_str: str) -> float:
    """Parse score string to float"""
    if not score_str or score_str in ["Not Found", "N/A", ""]:
        return 0.0
    try:
        clean = re.sub(r'[^\d.]', '', str(score_str))
        return float(clean) if clean else 0.0
    except:
        return 0.0

def get_score_color(score: float) -> str:
    """Get color based on score"""
    if score >= 80:
        return "#00C851"  # Green
    elif score >= 60:
        return "#ffbb33"  # Yellow
    elif score >= 40:
        return "#ff8800"  # Orange
    else:
        return "#ff4444"  # Red

def extract_asin(asin_str: str) -> str:
    """Extract clean ASIN from hyperlink or string"""
    if not asin_str:
        return ""
    match = re.search(r'([A-Z0-9]{10})', str(asin_str))
    return match.group(1) if match else str(asin_str)[:10]

def create_amazon_link(asin: str) -> str:
    """Create Amazon product link"""
    return f"https://www.amazon.com/dp/{asin}"

# ============================================
# 📈 VISUALIZATION FUNCTIONS
# ============================================
def create_score_radar_chart(scores: dict, title: str = "Оцінки листингу") -> go.Figure:
    """Create radar chart for scores"""
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Оцінка'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5),
        height=400,
        margin=dict(l=80, r=80, t=60, b=40)
    )
    
    return fig

def create_comparison_bar_chart(df: pd.DataFrame, metric_col: str, label_col: str, title: str) -> go.Figure:
    """Create horizontal bar chart for comparison"""
    fig = go.Figure()
    
    colors = [get_score_color(v) for v in df[metric_col]]
    
    fig.add_trace(go.Bar(
        y=df[label_col],
        x=df[metric_col],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in df[metric_col]],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 105], title="Оцінка %"),
        yaxis=dict(title=""),
        height=max(300, len(df) * 35),
        margin=dict(l=150, r=50, t=50, b=40)
    )
    
    return fig

def create_benchmarking_chart(df_bench: pd.DataFrame) -> go.Figure:
    """Create benchmarking comparison chart"""
    if df_bench.empty:
        return None
    
    fig = go.Figure()
    
    # Extract our scores and competitor scores
    criteria = df_bench['Критерій'].tolist() if 'Критерій' in df_bench.columns else []
    our_scores = []
    comp_scores = []
    
    for _, row in df_bench.iterrows():
        our_val = parse_score(str(row.get('Мы (Our %)', '0')))
        comp_val = parse_score(str(row.get('Конк #1 (%)', '0')))
        our_scores.append(our_val)
        comp_scores.append(comp_val)
    
    fig.add_trace(go.Bar(
        name='🏠 Наші товари',
        x=criteria,
        y=our_scores,
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='🎯 Конкуренти',
        x=criteria,
        y=comp_scores,
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="📊 Порівняння: Ми vs Конкуренти",
        barmode='group',
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="Оцінка %", range=[0, 105]),
        height=500,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    
    return fig

# ============================================
# 🎯 MAIN DASHBOARD
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Smart Listing AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Аналіз та оптимізація Amazon листингів під AI-агентів (Rufus, Cosmo)**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/amazon.png", width=80)
        st.markdown("### ⚙️ Налаштування")
        
        # Language selector
        lang = st.selectbox("🌐 Мова", ["UA", "RU", "EN"], index=0)
        
        # Refresh button
        if st.button("🔄 Оновити дані", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Auto-update info
        st.markdown("### 🔄 Автооновлення")
        st.caption("Дані оновлюються автоматично")
        st.caption("Скрипт працює 24/7 на сервері")
        
        # Last update time from data
        df_check = load_sheet_data("Listing Analysis")
        if not df_check.empty and 'Дата анализа' in df_check.columns:
            last_date = df_check['Дата анализа'].iloc[-1] if len(df_check) > 0 else "N/A"
            st.caption(f"Останнє оновлення: {last_date}")
        
        st.markdown("---")
        
        # Config info
        config = load_config()
        if config:
            st.markdown("### 📋 Поточна конфігурація")
            
            # Count ASINs
            product_urls = config.get("product_urls", "")
            competitor_urls = config.get("competitor_urls", "")
            
            product_count = len([a for a in product_urls.split(",") if a.strip()]) if product_urls else 0
            competitor_count = len([a for a in competitor_urls.split(",") if a.strip()]) if competitor_urls else 0
            
            st.metric("🏠 Наші ASIN", product_count)
            st.metric("🎯 Конкуренти", competitor_count)
            
            st.markdown("---")
            
            # Model info
            st.markdown("### 🤖 Моделі")
            st.caption(f"LITE: {config.get('LITE_MODEL', 'N/A')}")
            st.caption(f"POWER: {config.get('POWER_MODEL', 'N/A')}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Огляд",
        "📈 Listing Analysis", 
        "🏆 Benchmarking",
        "🛠️ Optimization",
        "⚙️ Управління ASIN"
    ])
    
    # ========================================
    # TAB 1: OVERVIEW
    # ========================================
    with tab1:
        st.markdown("## 📊 Загальний огляд")
        
        # Load data
        df_analysis = load_sheet_data("Listing Analysis")
        df_bench = load_sheet_data("Benchmarking")
        
        if df_analysis.empty:
            st.warning("⚠️ Дані аналізу не знайдено. Запустіть аналіз спочатку.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_products = len(df_analysis)
            
            # Calculate average overall score
            if 'Загальна оцінка' in df_analysis.columns:
                avg_score = df_analysis['Загальна оцінка'].apply(parse_score).mean()
            elif 'Общая оценка' in df_analysis.columns:
                avg_score = df_analysis['Общая оценка'].apply(parse_score).mean()
            else:
                avg_score = 0
            
            # Count by type
            own_count = len(df_analysis[df_analysis.get('Тип', pd.Series()) == 'Собственный']) if 'Тип' in df_analysis.columns else 0
            comp_count = len(df_analysis[df_analysis.get('Тип', pd.Series()) == 'Конкурент']) if 'Тип' in df_analysis.columns else 0
            
            with col1:
                st.metric("📦 Всього товарів", total_products)
            
            with col2:
                st.metric("🏠 Наші", own_count)
            
            with col3:
                st.metric("🎯 Конкуренти", comp_count)
            
            with col4:
                delta_color = "normal" if avg_score >= 70 else "inverse"
                st.metric("📊 Середня оцінка", f"{avg_score:.1f}%", delta=f"{'✅' if avg_score >= 70 else '⚠️'}")
            
            st.markdown("---")
            
            # Quick comparison chart
            if not df_analysis.empty and 'Тип' in df_analysis.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score distribution
                    score_col = 'Загальна оцінка' if 'Загальна оцінка' in df_analysis.columns else 'Общая оценка'
                    if score_col in df_analysis.columns:
                        df_plot = df_analysis.copy()
                        df_plot['Score'] = df_plot[score_col].apply(parse_score)
                        df_plot['ASIN_clean'] = df_plot['ASIN'].apply(extract_asin)
                        
                        fig = px.bar(
                            df_plot, 
                            x='ASIN_clean', 
                            y='Score',
                            color='Тип',
                            color_discrete_map={'Собственный': '#667eea', 'Конкурент': '#ff6b6b'},
                            title="📊 Загальні оцінки по ASIN"
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top/Bottom performers
                    if score_col in df_analysis.columns:
                        df_sorted = df_plot.sort_values('Score', ascending=False)
                        
                        st.markdown("#### 🏆 Топ-5 найкращих")
                        for i, row in df_sorted.head(5).iterrows():
                            asin = extract_asin(row['ASIN'])
                            score = row['Score']
                            typ = row.get('Тип', 'N/A')
                            emoji = "🏠" if typ == "Собственный" else "🎯"
                            color = get_score_color(score)
                            st.markdown(f"{emoji} [{asin}]({create_amazon_link(asin)}) - <span style='color:{color}'>{score:.1f}%</span>", unsafe_allow_html=True)
                        
                        st.markdown("#### ⚠️ Потребують уваги")
                        for i, row in df_sorted.tail(3).iterrows():
                            asin = extract_asin(row['ASIN'])
                            score = row['Score']
                            typ = row.get('Тип', 'N/A')
                            emoji = "🏠" if typ == "Собственный" else "🎯"
                            color = get_score_color(score)
                            st.markdown(f"{emoji} [{asin}]({create_amazon_link(asin)}) - <span style='color:{color}'>{score:.1f}%</span>", unsafe_allow_html=True)
    
    # ========================================
    # TAB 2: LISTING ANALYSIS
    # ========================================
    with tab2:
        st.markdown("## 📈 Детальний аналіз листингів")
        
        df_analysis = load_sheet_data("Listing Analysis")
        
        if df_analysis.empty:
            st.warning("⚠️ Дані аналізу не знайдено.")
        else:
            # ASIN selector
            asin_list = df_analysis['ASIN'].apply(extract_asin).tolist()
            selected_asin = st.selectbox("🔍 Виберіть ASIN для детального аналізу", asin_list)
            
            if selected_asin:
                # Filter data for selected ASIN
                row = df_analysis[df_analysis['ASIN'].apply(extract_asin) == selected_asin].iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### 📦 {selected_asin}")
                    st.markdown(f"[🔗 Відкрити на Amazon]({create_amazon_link(selected_asin)})")
                    
                    typ = row.get('Тип', 'N/A')
                    st.markdown(f"**Тип:** {'🏠 Власний' if typ == 'Собственный' else '🎯 Конкурент'}")
                    
                    brand = row.get('Бренд', 'N/A')
                    st.markdown(f"**Бренд:** {brand}")
                    
                    # Show title
                    title = row.get('Название товара', row.get('Заголовок (Title)', 'N/A'))
                    if title and len(str(title)) > 5:
                        with st.expander("📝 Заголовок"):
                            st.write(title)
                
                with col2:
                    # Radar chart with scores
                    score_mapping = {
                        'Заголовок': 'Оценка заголовка',
                        'Буллети': 'Оценка буллетов',
                        'Опис': 'Оценка описания',
                        'Зображення': 'Оценка изображений',
                        'Q&A': 'Оценка Q&A',
                        'Відгуки': 'Оценка отзывов',
                        'A+': 'Оценка A+ контента',
                        'Ціна': 'Оценка цены',
                        'Keywords': 'Оценка ключевых слов'
                    }
                    
                    scores = {}
                    for label, col_name in score_mapping.items():
                        if col_name in row.index:
                            scores[label] = parse_score(str(row[col_name]))
                    
                    if scores:
                        fig = create_score_radar_chart(scores, f"Оцінки {selected_asin}")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed scores table
                st.markdown("### 📋 Детальні оцінки")
                
                all_score_cols = [col for col in row.index if 'оценка' in col.lower() or 'score' in col.lower()]
                
                if all_score_cols:
                    scores_data = []
                    for col in all_score_cols:
                        score_val = parse_score(str(row[col]))
                        scores_data.append({
                            'Параметр': col.replace('Оценка ', '').replace('_score', ''),
                            'Оцінка': f"{score_val:.1f}%",
                            'Статус': '✅' if score_val >= 80 else '⚠️' if score_val >= 60 else '❌'
                        })
                    
                    df_scores = pd.DataFrame(scores_data)
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)
    
    # ========================================
    # TAB 3: BENCHMARKING
    # ========================================
    with tab3:
        st.markdown("## 🏆 Бенчмаркінг: Ми vs Конкуренти")
        
        df_bench = load_sheet_data("Benchmarking")
        
        if df_bench.empty:
            st.warning("⚠️ Дані бенчмаркінгу не знайдено.")
        else:
            # Filter out summary rows
            df_bench_filtered = df_bench[~df_bench['Критерій'].str.contains('СТАТИСТИКА|ИТОГО|📊', na=False, case=False)]
            
            if not df_bench_filtered.empty:
                # Create comparison chart
                fig = create_benchmarking_chart(df_bench_filtered)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                st.markdown("### 📊 Підсумок")
                
                col1, col2, col3 = st.columns(3)
                
                # Calculate wins/losses
                wins = 0
                losses = 0
                
                for _, row in df_bench_filtered.iterrows():
                    our = parse_score(str(row.get('Мы (Our %)', '0')))
                    comp = parse_score(str(row.get('Конк #1 (%)', '0')))
                    if our > comp:
                        wins += 1
                    elif comp > our:
                        losses += 1
                
                with col1:
                    st.metric("🏆 Ми виграємо", f"{wins} критеріїв")
                
                with col2:
                    st.metric("😔 Ми програємо", f"{losses} критеріїв")
                
                with col3:
                    total = wins + losses
                    win_rate = (wins / total * 100) if total > 0 else 0
                    st.metric("📈 Win Rate", f"{win_rate:.1f}%")
                
                # Detailed table
                st.markdown("### 📋 Детальна таблиця")
                st.dataframe(df_bench_filtered, use_container_width=True, hide_index=True)
    
    # ========================================
    # TAB 4: OPTIMIZATION
    # ========================================
    with tab4:
        st.markdown("## 🛠️ Рекомендації з оптимізації")
        
        df_opt = load_sheet_data("Listing Optimization AI")
        
        if df_opt.empty:
            st.warning("⚠️ Дані оптимізації не знайдено.")
        else:
            # ASIN selector
            asin_list = df_opt['ASIN'].apply(extract_asin).tolist() if 'ASIN' in df_opt.columns else []
            
            if asin_list:
                selected_asin = st.selectbox("🔍 Виберіть ASIN", asin_list, key="opt_asin")
                
                if selected_asin:
                    row = df_opt[df_opt['ASIN'].apply(extract_asin) == selected_asin].iloc[0]
                    
                    st.markdown(f"### 📦 Рекомендації для [{selected_asin}]({create_amazon_link(selected_asin)})")
                    
                    # Title optimization
                    with st.expander("📝 Заголовок (Title)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінал:**")
                            orig_title = row.get('Оригинальный Title', row.get('Заголовок (Title)', 'N/A'))
                            st.text_area("", orig_title, height=100, key="orig_title", disabled=True)
                        with col2:
                            st.markdown("**Оптимізований:**")
                            opt_title = row.get('Оптимизированный Title', 'N/A')
                            st.text_area("", opt_title, height=100, key="opt_title", disabled=True)
                        
                        rationale = row.get('Рекомендации и улучшения Title', row.get('Рекомендації Title', ''))
                        if rationale:
                            st.info(f"💡 {rationale}")
                    
                    # Bullets optimization
                    with st.expander("🔹 Буллети (Feature Bullets)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінал:**")
                            orig_bullets = row.get('Оригинальные Bullets', 'N/A')
                            st.text_area("", str(orig_bullets)[:2000], height=200, key="orig_bullets", disabled=True)
                        with col2:
                            st.markdown("**Оптимізовані:**")
                            opt_bullets = row.get('Оптимизированные Bullets', 'N/A')
                            st.text_area("", str(opt_bullets)[:2000], height=200, key="opt_bullets", disabled=True)
                    
                    # Images recommendations
                    with st.expander("📸 Зображення"):
                        img_analysis = row.get('Анализ изображений', row.get('AI анализ изображений', ''))
                        img_recs = row.get('Рекомендации по изображениям', '')
                        
                        if img_analysis:
                            st.markdown("**AI Аналіз:**")
                            st.text_area("", str(img_analysis)[:3000], height=200, key="img_analysis", disabled=True)
                        
                        if img_recs:
                            st.markdown("**Рекомендації:**")
                            st.info(img_recs)
                    
                    # Keywords
                    with st.expander("🔑 Ключові слова"):
                        orig_kw = row.get('Оригинальные Keywords', 'N/A')
                        opt_kw = row.get('Оптимизированные Keywords', 'N/A')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінальні:**")
                            st.text_area("", str(orig_kw)[:1500], height=150, key="orig_kw", disabled=True)
                        with col2:
                            st.markdown("**Оптимізовані:**")
                            st.text_area("", str(opt_kw)[:1500], height=150, key="opt_kw", disabled=True)
                    
                    # General recommendations
                    general = row.get('Общий комментарий по оптимизации', '')
                    if general:
                        st.markdown("### 💡 Загальні рекомендації")
                        st.success(general)
    
    # Footer
    st.markdown("---")
    
    # ========================================
    # TAB 5: ASIN MANAGEMENT
    # ========================================
    with tab5:
        st.markdown("## ⚙️ Управління ASIN")
        
        st.info("""
        🔄 **Як це працює:**
        - Введіть ASIN тут → вони зберігаються в Google Sheets Config
        - Скрипт на сервері автоматично підхоплює нові ASIN
        - Результати з'являються в Dashboard через 5-10 хвилин
        """)
        
        # Load current config (fresh, no cache)
        current_config = load_config_fresh()
        
        # Parse current ASINs
        current_products = current_config.get("product_urls", "")
        current_competitors = current_config.get("competitor_urls", "")
        
        # Extract ASINs from URLs
        def extract_asins_from_urls(urls_str: str) -> list:
            """Extract ASINs from URL string"""
            if not urls_str:
                return []
            asins = []
            # Split by common delimiters
            parts = urls_str.replace('\n', ',').replace('__', ',').split(',')
            for part in parts:
                match = re.search(r'([A-Z0-9]{10})', part.strip())
                if match:
                    asins.append(match.group(1))
            return list(set(asins))  # Remove duplicates
        
        product_asins = extract_asins_from_urls(current_products)
        competitor_asins = extract_asins_from_urls(current_competitors)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏠 Наші ASIN")
            st.caption(f"Поточна кількість: {len(product_asins)}")
            
            # Show current ASINs
            product_text = st.text_area(
                "ASIN (кожен з нового рядка або через кому)",
                value="\n".join(product_asins),
                height=200,
                key="edit_product_asins",
                help="Введіть ASIN товарів для моніторингу"
            )
            
            # Category for products
            current_cat_product = current_config.get("Category_product", "men ss")
            cat_product = st.text_input(
                "Категорія (наші)",
                value=current_cat_product,
                key="cat_product"
            )
        
        with col2:
            st.markdown("### 🎯 ASIN Конкурентів")
            st.caption(f"Поточна кількість: {len(competitor_asins)}")
            
            # Show current competitor ASINs
            competitor_text = st.text_area(
                "ASIN конкурентів",
                value="\n".join(competitor_asins),
                height=200,
                key="edit_competitor_asins",
                help="Введіть ASIN конкурентів для порівняння"
            )
            
            # Category for competitors
            current_cat_competitor = current_config.get("Category_competitor", "men ss")
            cat_competitor = st.text_input(
                "Категорія (конкуренти)",
                value=current_cat_competitor,
                key="cat_competitor"
            )
        
        st.markdown("---")
        
        # Save button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Зберегти зміни в Config", use_container_width=True, type="primary"):
                # Parse new ASINs
                new_products = [a.strip() for a in product_text.replace(',', '\n').split('\n') if a.strip() and len(a.strip()) == 10]
                new_competitors = [a.strip() for a in competitor_text.replace(',', '\n').split('\n') if a.strip() and len(a.strip()) == 10]
                
                # Format as Amazon URLs with __ separator (як в твоєму Config)
                product_urls_formatted = "__".join([f"https://www.amazon.com/dp/{asin}" for asin in new_products]) if new_products else ""
                competitor_urls_formatted = "__".join([f"https://www.amazon.com/dp/{asin}" for asin in new_competitors]) if new_competitors else ""
                
                # Save to Config
                success = True
                
                if save_to_config("product_urls", product_urls_formatted):
                    st.success(f"✅ Збережено {len(new_products)} наших ASIN")
                else:
                    success = False
                
                if save_to_config("competitor_urls", competitor_urls_formatted):
                    st.success(f"✅ Збережено {len(new_competitors)} ASIN конкурентів")
                else:
                    success = False
                
                if save_to_config("Category_product", cat_product):
                    pass
                else:
                    success = False
                    
                if save_to_config("Category_competitor", cat_competitor):
                    pass
                else:
                    success = False
                
                if success:
                    st.balloons()
                    st.success("🎉 Всі зміни збережено! Скрипт підхопить їх автоматично.")
                    # Clear cache to show updated data
                    st.cache_data.clear()
        
        # Quick add section
        st.markdown("---")
        st.markdown("### ➕ Швидке додавання ASIN")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            quick_asin = st.text_input(
                "Введіть ASIN",
                placeholder="B08HSD4FNW",
                key="quick_add_asin"
            )
        
        with col2:
            asin_type = st.selectbox(
                "Тип",
                ["🏠 Наш", "🎯 Конкурент"],
                key="quick_add_type"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Додати", key="quick_add_btn"):
                if quick_asin and len(quick_asin) == 10:
                    if "Наш" in asin_type:
                        if quick_asin not in product_asins:
                            product_asins.append(quick_asin)
                            product_urls_formatted = "__".join([f"https://www.amazon.com/dp/{asin}" for asin in product_asins])
                            if save_to_config("product_urls", product_urls_formatted):
                                st.success(f"✅ {quick_asin} додано до наших товарів!")
                                st.cache_data.clear()
                                st.rerun()
                        else:
                            st.warning(f"⚠️ {quick_asin} вже є в списку")
                    else:
                        if quick_asin not in competitor_asins:
                            competitor_asins.append(quick_asin)
                            competitor_urls_formatted = "__".join([f"https://www.amazon.com/dp/{asin}" for asin in competitor_asins])
                            if save_to_config("competitor_urls", competitor_urls_formatted):
                                st.success(f"✅ {quick_asin} додано до конкурентів!")
                                st.cache_data.clear()
                                st.rerun()
                        else:
                            st.warning(f"⚠️ {quick_asin} вже є в списку")
                else:
                    st.error("❌ Введіть коректний ASIN (10 символів)")
        
        # Preview links
        if quick_asin and len(quick_asin) >= 10:
            st.markdown(f"🔗 [Переглянути на Amazon](https://www.amazon.com/dp/{quick_asin[:10]})")
    
    # Footer bottom
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Аналіз працює автоматично 24/7")
    with col2:
        st.caption("📊 [Google Sheets](https://docs.google.com/spreadsheets/d/1_0WrdwdWthtaMHSAiNy8HqpAsTW9xNStTw7o9JDEWWU)")
    with col3:
        st.caption("Smart Listing AI v2.0 | Merino.tech")

if __name__ == "__main__":
    main()
