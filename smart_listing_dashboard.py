"""
🎯 Smart Listing AI Dashboard v2.2 - ГОТОВА ВЕРСІЯ
Streamlit-based visualization for Amazon Listing Analysis

✨ ЩО НОВОГО:
- Покращена синхронізація ASIN з Config
- Валідація ASIN при збереженні
- Діагностика Config в один клік
- Preview перед збереженням
- Детальні статуси операцій
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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
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
    .score-high { color: #00C851; font-weight: bold; }
    .score-medium { color: #ffbb33; font-weight: bold; }
    .score-low { color: #ff4444; font-weight: bold; }
    .asin-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
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

@st.cache_data(ttl=300)
def load_sheet_data(sheet_name: str) -> pd.DataFrame:
    """Generic loader for simple sheets"""
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

# ============================================
# 🆕 ПОКРАЩЕНІ ФУНКЦІЇ ДЛЯ CONFIG
# ============================================

def save_to_config(key: str, value: str, verbose: bool = False) -> bool:
    """Save value to Config sheet with validation"""
    try:
        creds = get_google_credentials()
        if not creds:
            return False
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("Config")
        
        all_data = worksheet.get_all_values()
        
        # Find the row with this key
        for i, row in enumerate(all_data):
            if row and row[0].strip() == key:
                worksheet.update_cell(i + 1, 2, value)
                if verbose:
                    st.success(f"✅ Оновлено рядок {i+1}, колонка B")
                return True
        
        # If key not found, append new row
        worksheet.append_row([key, value, ""])
        if verbose:
            st.info(f"ℹ️ Додано новий рядок для ключа '{key}'")
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
    """Load configuration from Config sheet (cached)"""
    return load_config_fresh()

def validate_asin(asin: str) -> bool:
    """Validate ASIN format"""
    asin = asin.strip().upper()
    return len(asin) == 10 and asin.isalnum()

def extract_asins_from_urls(urls_str: str) -> list:
    """Extract ASINs from URL string"""
    if not urls_str:
        return []
    asins = []
    parts = urls_str.split("__")
    for part in parts:
        match = re.search(r'([A-Z0-9]{10})', part.strip())
        if match:
            asin = match.group(1)
            if asin not in asins:
                asins.append(asin)
    return asins

def format_asins_for_config(asins: list[str]) -> str:
    """Format ASINs as Config value"""
    if not asins:
        return ""
    urls = [f"https://www.amazon.com/dp/{asin}" for asin in asins]
    return "__".join(urls)

def diagnose_config_sheet():
    """Run diagnostic on Config sheet"""
    st.markdown("### 🔍 Діагностика Config")
    
    try:
        creds = get_google_credentials()
        if not creds:
            st.error("❌ Помилка авторизації")
            return
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        
        try:
            worksheet = spreadsheet.worksheet("Config")
            st.success("✅ Лист 'Config' знайдено")
        except:
            st.error("❌ Лист 'Config' не існує!")
            return
        
        all_data = worksheet.get_all_values()
        st.write(f"📊 Всього рядків: {len(all_data)}")
        
        # Find headers
        header_row_idx = None
        for i, row in enumerate(all_data[:10]):
            if row and any("Key" in str(cell) or "key" in str(cell).lower() for cell in row):
                header_row_idx = i
                st.success(f"✅ Заголовки знайдено в рядку {i+1}: {row}")
                break
        
        if header_row_idx is None:
            st.warning("⚠️ Заголовки не знайдено")
        
        # Check ASIN keys
        st.markdown("#### Перевірка ASIN ключів:")
        
        for i, row in enumerate(all_data):
            if row and len(row) >= 2:
                key = row[0].strip()
                if key in ["product_urls", "competitor_urls"]:
                    value = row[1]
                    asins = re.findall(r'([A-Z0-9]{10})', value)
                    
                    st.success(f"✅ **{key}** (рядок {i+1})")
                    st.write(f"   - Знайдено ASIN'ів: {len(asins)}")
                    if len(asins) > 0:
                        with st.expander(f"Показати ASIN'и ({len(asins)})"):
                            for idx, asin in enumerate(asins, 1):
                                st.write(f"   {idx}. {asin}")
        
    except Exception as e:
        st.error(f"❌ Помилка: {e}")

@st.cache_data(ttl=300)
def load_benchmarking_data() -> pd.DataFrame:
    """Load Benchmarking sheet"""
    try:
        creds = get_google_credentials()
        if not creds:
            return pd.DataFrame()

        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("Benchmarking")

        raw = worksheet.get_all_values()
        if not raw:
            return pd.DataFrame()

        header_row_index = None
        for i, row in enumerate(raw):
            row_lower = [c.strip().lower() for c in row if c is not None]
            if any(col in row_lower for col in ["критерий", "критерій"]):
                header_row_index = i
                break

        if header_row_index is None:
            return pd.DataFrame()

        headers = raw[header_row_index]
        data_rows = raw[header_row_index + 1:]

        df = pd.DataFrame(data_rows, columns=headers)

        first_col = headers[0]
        df[first_col] = df[first_col].astype(str)
        df = df[df[first_col].str.strip() != ""]

        return df

    except Exception as e:
        st.error(f"❌ Ошибка загрузки Benchmarking: {e}")
        return pd.DataFrame()

# ============================================
# 🤖 PROMPTS
# ============================================

def get_prompt_by_id(prompt_id: str, sheet_name: str) -> dict:
    """Load a specific prompt by ID from sheet"""
    try:
        creds = get_google_credentials()
        if not creds:
            return {}
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        raw_data = worksheet.get_all_values()
        if not raw_data or len(raw_data) < 2:
            return {}
        
        header_row_idx = None
        for idx, row in enumerate(raw_data[:5]):
            row_str = ' '.join([str(c).lower() for c in row])
            if 'назв' in row_str or 'промт' in row_str or 'system' in row_str:
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            header_row_idx = 0
        
        headers = raw_data[header_row_idx]
        
        id_col_idx = 0
        name_col_idx = None
        system_col_idx = None
        
        for i, h in enumerate(headers):
            h_lower = str(h).lower().strip()
            if 'назв' in h_lower or 'name' in h_lower or 'название' in h_lower:
                name_col_idx = i
            if 'system' in h_lower or 'промт' in h_lower or 'prompt' in h_lower:
                system_col_idx = i
        
        if name_col_idx is None and len(headers) > 1:
            name_col_idx = 1
        if system_col_idx is None:
            if len(headers) > 2:
                system_col_idx = 2
            else:
                system_col_idx = len(headers) - 1
        
        for row_idx, row in enumerate(raw_data[header_row_idx + 1:], start=header_row_idx + 1):
            if not row or len(row) == 0:
                continue
            
            row_id = str(row[id_col_idx]).strip()
            
            if row_id == prompt_id or prompt_id in row_id:
                result = {
                    "id": prompt_id,
                    "name": "",
                    "system_prompt": ""
                }
                
                if name_col_idx is not None and len(row) > name_col_idx:
                    result["name"] = str(row[name_col_idx]).strip()
                
                if system_col_idx is not None and len(row) > system_col_idx:
                    result["system_prompt"] = str(row[system_col_idx]).strip()
                
                return result
        
        return {}
        
    except Exception as e:
        st.error(f"❌ Помилка завантаження промта {prompt_id}: {e}")
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
        return "#00C851"
    elif score >= 60:
        return "#ffbb33"
    elif score >= 40:
        return "#ff8800"
    else:
        return "#ff4444"

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
    
    if categories:
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

def create_benchmarking_chart(df_bench: pd.DataFrame) -> go.Figure | None:
    """Create benchmarking comparison chart"""
    if df_bench.empty:
        return None

    crit_col = None
    if "Критерій" in df_bench.columns:
        crit_col = "Критерій"
    elif "Критерий" in df_bench.columns:
        crit_col = "Критерий"
    else:
        return None

    mask = ~df_bench[crit_col].astype(str).str.contains("СТАТИСТИКА|ИТОГ|ИТОГО|📊", na=False, case=False)
    df_bench_filtered = df_bench[mask].copy()

    if df_bench_filtered.empty:
        return None
    
    fig = go.Figure()
    
    criteria = df_bench_filtered[crit_col].tolist()
    our_scores = []
    comp_scores = []
    
    for _, row in df_bench_filtered.iterrows():
        our_val = parse_score(str(row.get("Мы (Our %)", "0")))
        comp_val = parse_score(str(row.get("Конк #1 (%)", "0")))
        our_scores.append(our_val)
        comp_scores.append(comp_val)
    
    fig.add_trace(go.Bar(
        name="🏠 Наші товари",
        x=criteria,
        y=our_scores,
        marker_color="#667eea"
    ))
    
    fig.add_trace(go.Bar(
        name="🎯 Конкуренти",
        x=criteria,
        y=comp_scores,
        marker_color="#ff6b6b"
    ))
    
    fig.update_layout(
        title="📊 Порівняння: Ми vs Конкуренти",
        barmode="group",
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
        
        lang = st.selectbox("🌐 Мова", ["UA", "RU", "EN"], index=0)
        
        if st.button("🔄 Оновити дані", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🔄 Автооновлення")
        st.caption("Дані оновлюються автоматично")
        st.caption("Скрипт працює 24/7 на сервері")
        
        df_check = load_sheet_data("Listing Analysis")
        if not df_check.empty and "Дата анализа" in df_check.columns:
            last_date = df_check["Дата анализа"].iloc[-1] if len(df_check) > 0 else "N/A"
            st.caption(f"Останнє оновлення: {last_date}")
        
        st.markdown("---")
        
        config = load_config()
        if config:
            st.markdown("### 📋 Поточна конфігурація")
            
            product_urls = config.get("product_urls", "")
            competitor_urls = config.get("competitor_urls", "")
            
            product_count = len([a for a in product_urls.split(",") if a.strip()]) if product_urls else 0
            competitor_count = len([a for a in competitor_urls.split(",") if a.strip()]) if competitor_urls else 0
            
            st.metric("🏠 Наші ASIN", product_count)
            st.metric("🎯 Конкуренти", competitor_count)
            
            st.markdown("---")
            
            st.markdown("### 🤖 Моделі")
            st.caption(f"LITE: {config.get('LITE_MODEL', 'N/A')}")
            st.caption(f"POWER: {config.get('POWER_MODEL', 'N/A')}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Огляд",
        "📈 Аналіз листингів", 
        "🏆 Бенчмаркінг",
        "🛠️ Оптимізація",
        "⚙️ Управління ASIN",
        "✏️ Редактор промтів"
    ])
    
    # ========================================
    # TAB 1: OVERVIEW
    # ========================================
    with tab1:
        st.markdown("## 📊 Загальний огляд")
        
        df_analysis = load_sheet_data("Listing Analysis")
        df_bench = load_benchmarking_data()
        
        if df_analysis.empty:
            st.warning("⚠️ Дані аналізу не знайдено. Запустіть аналіз спочатку.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            total_products = len(df_analysis)
            
            if "Загальна оцінка" in df_analysis.columns:
                avg_score = df_analysis["Загальна оцінка"].apply(parse_score).mean()
            elif "Общая оценка" in df_analysis.columns:
                avg_score = df_analysis["Общая оценка"].apply(parse_score).mean()
            else:
                avg_score = 0.0
            
            own_count = len(df_analysis[df_analysis.get("Тип", pd.Series()) == "Собственный"]) if "Тип" in df_analysis.columns else 0
            comp_count = len(df_analysis[df_analysis.get("Тип", pd.Series()) == "Конкурент"]) if "Тип" in df_analysis.columns else 0
            
            with col1:
                st.metric("📦 Всього товарів", total_products)
            
            with col2:
                st.metric("🏠 Наші", own_count)
            
            with col3:
                st.metric("🎯 Конкуренти", comp_count)
            
            with col4:
                st.metric("📊 Середня оцінка", f"{avg_score:.1f}%", delta=f"{'✅' if avg_score >= 70 else '⚠️'}")
            
            st.markdown("---")
            
            if not df_analysis.empty and "Тип" in df_analysis.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    score_col = "Загальна оцінка" if "Загальна оцінка" in df_analysis.columns else "Общая оценка"
                    if score_col in df_analysis.columns:
                        df_plot = df_analysis.copy()
                        df_plot["Score"] = df_plot[score_col].apply(parse_score)
                        df_plot["ASIN_clean"] = df_plot["ASIN"].apply(extract_asin)
                        
                        fig = px.bar(
                            df_plot, 
                            x="ASIN_clean", 
                            y="Score",
                            color="Тип",
                            color_discrete_map={"Собственный": "#667eea", "Конкурент": "#ff6b6b"},
                            title="📊 Загальні оцінки по ASIN"
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if score_col in df_analysis.columns:
                        df_sorted = df_plot.sort_values("Score", ascending=False)
                        
                        st.markdown("#### 🏆 Топ-5 найкращих")
                        for _, row in df_sorted.head(5).iterrows():
                            asin = extract_asin(row["ASIN"])
                            score = row["Score"]
                            typ = row.get("Тип", "N/A")
                            emoji = "🏠" if typ == "Собственный" else "🎯"
                            color = get_score_color(score)
                            st.markdown(
                                f"{emoji} [{asin}]({create_amazon_link(asin)}) - "
                                f"<span style='color:{color}'>{score:.1f}%</span>",
                                unsafe_allow_html=True
                            )
                        
                        st.markdown("#### ⚠️ Потребують уваги")
                        for _, row in df_sorted.tail(3).iterrows():
                            asin = extract_asin(row["ASIN"])
                            score = row["Score"]
                            typ = row.get("Тип", "N/A")
                            emoji = "🏠" if typ == "Собственный" else "🎯"
                            color = get_score_color(score)
                            st.markdown(
                                f"{emoji} [{asin}]({create_amazon_link(asin)}) - "
                                f"<span style='color:{color}'>{score:.1f}%</span>",
                                unsafe_allow_html=True
                            )
    
    # ========================================
    # TAB 2: LISTING ANALYSIS
    # ========================================
    with tab2:
        st.markdown("## 📈 Детальний аналіз листингів")
        
        df_analysis = load_sheet_data("Listing Analysis")
        
        if df_analysis.empty:
            st.warning("⚠️ Дані аналізу не знайдено.")
        else:
            asin_list = df_analysis["ASIN"].apply(extract_asin).tolist()
            selected_asin = st.selectbox("🔍 Виберіть ASIN для детального аналізу", asin_list)
            
            if selected_asin:
                row = df_analysis[df_analysis["ASIN"].apply(extract_asin) == selected_asin].iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### 📦 {selected_asin}")
                    st.markdown(f"[🔗 Відкрити на Amazon]({create_amazon_link(selected_asin)})")
                    
                    typ = row.get("Тип", "N/A")
                    st.markdown(f"**Тип:** {'🏠 Власний' if typ == 'Собственный' else '🎯 Конкурент'}")
                    
                    brand = row.get("Бренд", "N/A")
                    st.markdown(f"**Бренд:** {brand}")
                    
                    title = row.get("Название товара", row.get("Заголовок (Title)", "N/A"))
                    if title and len(str(title)) > 5:
                        with st.expander("📝 Заголовок"):
                            st.write(title)
                
                with col2:
                    score_mapping = {
                        "Заголовок": "Оценка заголовка",
                        "Буллети": "Оценка буллетов",
                        "Опис": "Оценка описания",
                        "Зображення": "Оценка изображений",
                        "Q&A": "Оценка Q&A",
                        "Відгуки": "Оценка отзывов",
                        "A+": "Оценка A+ контента",
                        "Ціна": "Оценка цены",
                        "Keywords": "Оценка ключевых слов"
                    }
                    
                    scores = {}
                    for label, col_name in score_mapping.items():
                        if col_name in row.index:
                            scores[label] = parse_score(str(row[col_name]))
                    
                    if scores:
                        fig = create_score_radar_chart(scores, f"Оцінки {selected_asin}")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Детальні оцінки")
                
                all_score_cols = [col for col in row.index if "оценка" in col.lower() or "score" in col.lower()]
                
                if all_score_cols:
                    scores_data = []
                    for col in all_score_cols:
                        score_val = parse_score(str(row[col]))
                        scores_data.append({
                            "Параметр": col.replace("Оценка ", "").replace("_score", ""),
                            "Оцінка": f"{score_val:.1f}%",
                            "Статус": "✅" if score_val >= 80 else "⚠️" if score_val >= 60 else "❌"
                        })
                    
                    df_scores = pd.DataFrame(scores_data)
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### 🤖 Master Prompt PT000 (Listing Analysis)")
                pt000 = get_prompt_by_id("PT000", "Prompt Analysis")
                if not pt000:
                    st.error("PT000 не знайдено в листі 'Prompt Analysis'")
                else:
                    st.markdown(f"**ID:** `{pt000['id']}` &nbsp;&nbsp; **Назва:** {pt000['name']}")
                    st.text_area(
                        "System Prompt (read-only)",
                        pt000["system_prompt"],
                        height=350,
                        disabled=True
                    )
    
    # ========================================
    # TAB 3: BENCHMARKING
    # ========================================
    with tab3:
        st.markdown("## 🏆 Бенчмаркінг: Ми vs Конкуренти")
        
        df_bench = load_benchmarking_data()
        
        if df_bench.empty:
            st.warning("⚠️ Дані бенчмаркінгу не знайдено.")
        else:
            crit_col = None
            if "Критерій" in df_bench.columns:
                crit_col = "Критерій"
            elif "Критерий" in df_bench.columns:
                crit_col = "Критерий"

            if not crit_col:
                st.error("❌ В Benchmarking нет колонки 'Критерий' / 'Критерій'.")
            else:
                df_bench_filtered = df_bench[
                    ~df_bench[crit_col].astype(str).str.contains("СТАТИСТИКА|ИТОГ|ИТОГО|📊", na=False, case=False)
                ].copy()
                
                if not df_bench_filtered.empty:
                    fig = create_benchmarking_chart(df_bench_filtered)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📊 Підсумок")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    wins = 0
                    losses = 0
                    
                    for _, row in df_bench_filtered.iterrows():
                        our = parse_score(str(row.get("Мы (Our %)", "0")))
                        comp = parse_score(str(row.get("Конк #1 (%)", "0")))
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
            asin_list = df_opt["ASIN"].apply(extract_asin).tolist() if "ASIN" in df_opt.columns else []
            
            if asin_list:
                selected_asin = st.selectbox("🔍 Виберіть ASIN", asin_list, key="opt_asin")
                
                if selected_asin:
                    row = df_opt[df_opt["ASIN"].apply(extract_asin) == selected_asin].iloc[0]
                    
                    st.markdown(f"### 📦 Рекомендації для [{selected_asin}]({create_amazon_link(selected_asin)})")
                    
                    with st.expander("📝 Заголовок (Title)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінал:**")
                            orig_title = row.get("Оригинальный Title", row.get("Заголовок (Title)", "N/A"))
                            st.text_area("", orig_title, height=100, key="orig_title", disabled=True)
                        with col2:
                            st.markdown("**Оптимізований:**")
                            opt_title = row.get("Оптимизированный Title", "N/A")
                            st.text_area("", opt_title, height=100, key="opt_title", disabled=True)
                        
                        rationale = row.get("Рекомендации и улучшения Title", row.get("Рекомендації Title", ""))
                        if rationale:
                            st.info(f"💡 {rationale}")
                    
                    with st.expander("🔹 Буллети (Feature Bullets)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінал:**")
                            orig_bullets = row.get("Оригинальные Bullets", "N/A")
                            st.text_area("", str(orig_bullets)[:2000], height=200, key="orig_bullets", disabled=True)
                        with col2:
                            st.markdown("**Оптимізовані:**")
                            opt_bullets = row.get("Оптимизированные Bullets", "N/A")
                            st.text_area("", str(opt_bullets)[:2000], height=200, key="opt_bullets", disabled=True)
                    
                    with st.expander("📸 Зображення"):
                        img_analysis = row.get("Анализ изображений", row.get("AI анализ изображений", ""))
                        img_recs = row.get("Рекомендации по изображениям", "")
                        
                        if img_analysis:
                            st.markdown("**AI Аналіз:**")
                            st.text_area("", str(img_analysis)[:3000], height=200, key="img_analysis", disabled=True)
                        
                        if img_recs:
                            st.markdown("**Рекомендації:**")
                            st.info(img_recs)
                    
                    with st.expander("🔑 Ключові слова"):
                        orig_kw = row.get("Оригинальные Keywords", "N/A")
                        opt_kw = row.get("Оптимизированные Keywords", "N/A")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Оригінальні:**")
                            st.text_area("", str(orig_kw)[:1500], height=150, key="orig_kw", disabled=True)
                        with col2:
                            st.markdown("**Оптимізовані:**")
                            st.text_area("", str(opt_kw)[:1500], height=150, key="opt_kw", disabled=True)
                    
                    general = row.get("Общий комментарий по оптимизации", "")
                    if general:
                        st.markdown("### 💡 Загальні рекомендації")
                        st.success(general)

                st.markdown("---")
                st.markdown("### ✨ Master Prompt PT001 (Listing Optimization)")
                pt001 = get_prompt_by_id("PT001", "Prompt Optimization")
                if not pt001:
                    st.error("PT001 не знайдено в листі 'Prompt Optimization'")
                else:
                    st.markdown(f"**ID:** `{pt001['id']}` &nbsp;&nbsp; **Назва:** {pt001['name']}")
                    st.text_area(
                        "System Prompt (read-only)",
                        pt001["system_prompt"],
                        height=400,
                        disabled=True
                    )
    
    # ========================================
    # TAB 5: ASIN MANAGEMENT - ПОКРАЩЕНА ВЕРСІЯ
    # ========================================
    with tab5:
        st.markdown("## ⚙️ Управління ASIN")
        
        st.info("""
        🔄 **Як це працює:**
        - Введіть ASIN тут → вони зберігаються в Google Sheets **Config**
        - Ключі: `product_urls` та `competitor_urls`
        - Скрипт автоматично підхоплює нові ASIN
        - Результати з'являються через 5-10 хвилин
        """)
        
        # 🔍 ДІАГНОСТИКА
        with st.expander("🔍 Діагностика Config"):
            if st.button("▶️ Запустити діагностику"):
                diagnose_config_sheet()
        
        st.markdown("---")
        
        # Завантаження поточних даних
        current_config = load_config_fresh()
        
        current_products = current_config.get("product_urls", "")
        current_competitors = current_config.get("competitor_urls", "")
        
        product_asins = extract_asins_from_urls(current_products)
        competitor_asins = extract_asins_from_urls(current_competitors)
        
        # Статус
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏠 Наші ASIN (в Config)", len(product_asins))
        with col2:
            st.metric("🎯 Конкуренти (в Config)", len(competitor_asins))
        
        st.markdown("---")
        
        # Редактори
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏠 Наші ASIN")
            
            product_text = st.text_area(
                "ASIN (кожен з нового рядка)",
                value="\n".join(product_asins),
                height=200,
                key="edit_product_asins",
                help="10 символів, великі літери та цифри"
            )
            
            current_cat_product = current_config.get("Category_product", "")
            cat_product = st.text_input(
                "📁 Категорія",
                value=current_cat_product,
                key="cat_product"
            )
        
        with col2:
            st.markdown("### 🎯 ASIN Конкурентів")
            
            competitor_text = st.text_area(
                "ASIN конкурентів",
                value="\n".join(competitor_asins),
                height=200,
                key="edit_competitor_asins"
            )
            
            current_cat_competitor = current_config.get("Category_competitor", "")
            cat_competitor = st.text_input(
                "📁 Категорія",
                value=current_cat_competitor,
                key="cat_competitor"
            )
        
        st.markdown("---")
        
        # SAVE BUTTON
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Зберегти в Config", use_container_width=True, type="primary"):
                with st.spinner("🔄 Зберігаю..."):
                    # Валідація
                    new_products = []
                    invalid_products = []
                    
                    for line in product_text.replace(",", "\n").split("\n"):
                        asin = line.strip().upper()
                        if not asin:
                            continue
                        if validate_asin(asin):
                            new_products.append(asin)
                        else:
                            invalid_products.append(asin)
                    
                    new_competitors = []
                    invalid_competitors = []
                    
                    for line in competitor_text.replace(",", "\n").split("\n"):
                        asin = line.strip().upper()
                        if not asin:
                            continue
                        if validate_asin(asin):
                            new_competitors.append(asin)
                        else:
                            invalid_competitors.append(asin)
                    
                    # Видалення дублікатів
                    new_products = list(dict.fromkeys(new_products))
                    new_competitors = list(dict.fromkeys(new_competitors))
                    
                    # Попередження про невалідні
                    if invalid_products:
                        st.warning(f"⚠️ Пропущено {len(invalid_products)} невалідних ASIN (наші)")
                    if invalid_competitors:
                        st.warning(f"⚠️ Пропущено {len(invalid_competitors)} невалідних ASIN (конкуренти)")
                    
                    # Форматування
                    product_urls_formatted = format_asins_for_config(new_products)
                    competitor_urls_formatted = format_asins_for_config(new_competitors)
                    
                    # Збереження
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
                        st.success("🎉 Всі зміни збережено в Config!")
                        st.cache_data.clear()
                        import time
                        time.sleep(1)
                        st.rerun()
        
        # QUICK ADD
        st.markdown("---")
        st.markdown("### ➕ Швидке додавання")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            quick_asin = st.text_input(
                "ASIN",
                placeholder="B08HSD4FNW",
                key="quick_add_asin",
                max_chars=10
            )
        
        with col2:
            asin_type = st.selectbox(
                "Тип",
                ["🏠 Наш", "🎯 Конкурент"],
                key="quick_add_type"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            add_disabled = not quick_asin or not validate_asin(quick_asin)
            
            if st.button("➕ Додати", disabled=add_disabled):
                q = quick_asin.strip().upper()
                
                if "Наш" in asin_type:
                    if q not in product_asins:
                        product_asins.append(q)
                        formatted = format_asins_for_config(product_asins)
                        if save_to_config("product_urls", formatted):
                            st.success(f"✅ {q} додано!")
                            st.cache_data.clear()
                            st.rerun()
                    else:
                        st.warning(f"⚠️ {q} вже є")
                else:
                    if q not in competitor_asins:
                        competitor_asins.append(q)
                        formatted = format_asins_for_config(competitor_asins)
                        if save_to_config("competitor_urls", formatted):
                            st.success(f"✅ {q} додано!")
                            st.cache_data.clear()
                            st.rerun()
                    else:
                        st.warning(f"⚠️ {q} вже є")
        
        if quick_asin and len(quick_asin) >= 10:
            st.markdown(f"🔗 [Amazon](https://www.amazon.com/dp/{quick_asin[:10]})")
        
        # CURRENT STATE
        st.markdown("---")
        st.markdown("### 📊 Поточний список")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏠 Наші:**")
            if product_asins:
                for i, asin in enumerate(product_asins, 1):
                    st.write(f"{i}. [{asin}]({create_amazon_link(asin)})")
            else:
                st.caption("_Порожньо_")
        
        with col2:
            st.markdown("**🎯 Конкуренти:**")
            if competitor_asins:
                for i, asin in enumerate(competitor_asins, 1):
                    st.write(f"{i}. [{asin}]({create_amazon_link(asin)})")
            else:
                st.caption("_Порожньо_")
    
    # ========================================
    # TAB 6: PROMPT EDITOR
    # ========================================
    with tab6:
        st.markdown("## ✏️ Редактор промтів")
        st.caption("Редагуй master-промти PT000 та PT001")
        
        prompt_choice = st.radio(
            "🎯 Який промт редагувати?",
            [
                "PT000 - Listing Analysis",
                "PT001 - Listing Optimization"
            ],
            horizontal=False
        )
        
        if "PT000" in prompt_choice:
            sheet_name = "Prompt Analysis"
            prompt_id = "PT000"
        else:
            sheet_name = "Prompt Optimization"
            prompt_id = "PT001"
        
        st.markdown("---")
        
        prompt_data = get_prompt_by_id(prompt_id, sheet_name)
        
        if not prompt_data:
            st.error(f"❌ Промт {prompt_id} не знайдено в '{sheet_name}'")
        else:
            st.success(f"✅ {prompt_data['name']}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("ID", prompt_data['id'])
            with col2:
                st.metric("Лист", sheet_name)
            
            st.markdown("---")
            
            new_prompt = st.text_area(
                "🧠 System Prompt",
                value=prompt_data['system_prompt'],
                height=600,
                key=f"edit_{prompt_id}"
            )
            
            char_count = len(new_prompt)
            word_count = len(new_prompt.split())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📊 Символів: **{char_count:,}**")
            with col2:
                st.caption(f"📝 Слів: **{word_count:,}**")
            with col3:
                changed = new_prompt != prompt_data['system_prompt']
                st.caption(f"🔄 {'**Змінено ✏️**' if changed else 'Без змін ✅'}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "💾 Зберегти в Google Sheets", 
                    type="primary", 
                    use_container_width=True,
                    disabled=not changed
                ):
                    with st.spinner(f"🔄 Зберігаю {prompt_id}..."):
                        try:
                            creds = get_google_credentials()
                            client = gspread.authorize(creds)
                            spreadsheet = client.open_by_key(SPREADSHEET_ID)
                            ws = spreadsheet.worksheet(sheet_name)
                            
                            all_data = ws.get_all_values()
                            
                            header_row_idx = 0
                            for idx, row in enumerate(all_data[:5]):
                                row_str = ' '.join([str(c).lower() for c in row])
                                if 'назв' in row_str or 'промт' in row_str:
                                    header_row_idx = idx
                                    break
                            
                            headers = all_data[header_row_idx]
                            
                            id_col_idx = 0
                            system_col_idx = 2 if len(headers) > 2 else 1
                            
                            for i, h in enumerate(headers):
                                h_lower = str(h).lower().strip()
                                if 'system' in h_lower or ('промт' in h_lower and 'іі' in h_lower):
                                    system_col_idx = i
                            
                            target_row = None
                            for idx in range(header_row_idx + 1, len(all_data)):
                                if all_data[idx][id_col_idx].strip() == prompt_id:
                                    target_row = idx + 1
                                    break
                            
                            if target_row:
                                ws.update_cell(target_row, system_col_idx + 1, new_prompt)
                                st.success(f"✅ Промт {prompt_id} оновлено!")
                                st.balloons()
                                st.cache_data.clear()
                            else:
                                st.error(f"❌ Рядок з ID '{prompt_id}' не знайдено")
                        
                        except Exception as e:
                            st.error(f"❌ Помилка: {e}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Аналіз 24/7")
    with col2:
        st.caption("📊 [Google Sheets](https://docs.google.com/spreadsheets/d/1_0WrdwdWthtaMHSAiNy8HqpAsTW9xNStTw7o9JDEWWU)")
    with col3:
        st.caption("Smart Listing AI v2.2 | Merino.tech")


if __name__ == "__main__":
    main()

