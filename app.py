import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib
import warnings
import os

# ==================== 基础配置：忽略警告 + 中文/深色主题 ====================
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 永久解决中文显示问题 + 黑底白字图表主题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.facecolor': '#0E1117',
    'axes.facecolor': '#0E1117',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'grid.color': '#404040',
    'grid.alpha': 0.4,
    'legend.frameon': False,
})

# ==================== 核心函数：加载模型和数据（带容错处理） ====================
@st.cache_resource
def load_model():
    """加载机器学习模型和标签编码器"""
    try:
        model = joblib.load("models/xgb_final_predictor.pkl")
        le_gender = joblib.load("models/le_gender.pkl")
        le_major = joblib.load("models/le_major.pkl")
        return model, le_gender, le_major
    except FileNotFoundError:
        st.warning("⚠️ 模型文件未找到！成绩预测功能将不可用")
        return None, None, None
    except Exception as e:
        st.error(f"加载模型失败：{str(e)}")
        return None, None, None

@st.cache_data
def load_data():
    """加载学生成绩数据"""
    try:
        return pd.read_csv("student_data_adjusted_rounded.csv")
    except FileNotFoundError:
        st.warning("⚠️ 数据文件未找到！将使用模拟数据展示")
        # 生成模拟数据
        majors = ['大数据管理', '人工智能', '计算机科学', '软件工程', '信息安全']
        genders = ['男', '女']
        data = {
            '专业': np.random.choice(majors, 200),
            '性别': np.random.choice(genders, 200),
            '每周学习时长（小时）': np.random.uniform(5, 40, 200),
            '上课出勤率': np.random.uniform(0.6, 1.0, 200),
            '期中考试分数': np.random.uniform(50, 95, 200),
            '期末考试分数': np.random.uniform(50, 95, 200),
            '作业完成率': np.random.uniform(0.6, 1.0, 200),
            '学号': [f"2023{str(i).zfill(6)}" for i in range(1, 201)]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"加载数据失败：{str(e)}")
        return pd.DataFrame()

# 加载模型和数据
model, le_gender, le_major = load_model()
df = load_data()

# ==================== Streamlit 基础配置 ====================
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    layout="wide",
    page_icon="🎓"
)

# ==================== 自定义CSS样式（美化界面） ====================
st.markdown("""
<style>
/* ================== 全局背景 ================== */
.stApp {
    background: linear-gradient(180deg, #0B0F14 0%, #000000 100%);
    color: #E6E6E6;
}

/* ================== 侧边栏 ================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2A2A2A 0%, #1C1C1C 100%);
    border-right: 1px solid #333333;
}

/* 侧边栏内容文字 */
section[data-testid="stSidebar"] * {
    color: #DDDDDD;
    font-size: 15px;
}

/* 侧边栏标题 */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF;
    font-weight: 700;
}

/* ================== Radio / Select ================== */
div[data-baseweb="radio"] > div {
    background-color: #262626;
    border-radius: 12px;
    padding: 10px;
}

div[data-baseweb="radio"] label {
    padding: 6px 10px;
    border-radius: 8px;
    transition: all 0.25s ease;
}

/* Hover */
div[data-baseweb="radio"] label:hover {
    background-color: #333333;
}

/* 选中项 */
div[data-baseweb="radio"] input:checked + div {
    background: linear-gradient(135deg, #00C6FF, #0072FF);
    color: #FFFFFF;
    box-shadow: 0 0 12px rgba(0, 114, 255, 0.6);
}

/* ================== 按钮 ================== */
button {
    background: linear-gradient(135deg, #00C6FF, #0072FF);
    border-radius: 14px;
    border: none;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
}

button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(0, 114, 255, 0.4);
}

/* ================== 输入框 ================== */
input, textarea {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #333333 !important;
}

input:focus, textarea:focus {
    border-color: #00C6FF !important;
    box-shadow: 0 0 0 2px rgba(0,198,255,0.25) !important;
}

/* ================== Selectbox ================== */
div[data-baseweb="select"] {
    background-color: #1A1A1A;
    border-radius: 10px;
}

/* ================== Slider ================== */
div[data-baseweb="slider"] > div {
    color: white;
}

div[data-baseweb="slider"] div[role="slider"] {
    background: #00C6FF;
}

/* ================== 表格 DataFrame - 深黑主题 ================== */
[data-testid="stDataFrame"] {
    background-color: #000000 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stDataFrame"] .ag-root-wrapper,
[data-testid="stDataFrame"] .ag-body-viewport,
[data-testid="stDataFrame"] .ag-cell {
    background-color: #000000 !important;
    color: #e0e0e0 !important;
}

[data-testid="stDataFrame"] .ag-header {
    background-color: #0d1117 !important;
    border-bottom: 1px solid #1e1e1e !important;
}

[data-testid="stDataFrame"] .ag-header-cell-text {
    color: #ffffff !important;
    font-weight: 600;
}

/* 行背景 - 纯黑 + 轻微 hover 效果 */
[data-testid="stDataFrame"] .ag-row {
    background-color: #000000 !important;
}

[data-testid="stDataFrame"] .ag-row:hover {
    background-color: #1a1a1a !important;
}

/* 网格线 */
[data-testid="stDataFrame"] .ag-cell {
    border-color: #1e1e1e !important;
}

/* 选中行 */
[data-testid="stDataFrame"] .ag-row-selected {
    background-color: #0a2a4a !important;
}

/* 单元格文字强制 */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] .ag-cell-value {
    color: #f0f0f0 !important;
}

/* ================== 指标 Metric ================== */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161B22, #0D1117);
    padding: 18px;
    border-radius: 16px;
    box-shadow: inset 0 0 0 1px #222;
}

/* ================== Expander ================== */
details {
    background-color: #121212;
    border-radius: 14px;
    padding: 10px;
}

/* ================== 图片 ================== */
img {
    border-radius: 16px;
}

/* ================== 分割线 ================== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #333, transparent);
}

/* ================== 滚动条 ================== */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0B0F14;
}

::-webkit-scrollbar-thumb {
    background: #2E2E2E;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #00C6FF;
}

/* ================== 全局文字增强 ================== */
body, .stApp {
    color: #F2F2F2;
}

/* Markdown / 正文 */
.stMarkdown, .stText, .stWrite {
    color: #F0F0F0 !important;
    line-height: 1.75;
}

/* 标题 */
h1, h2, h3 {
    color: #FFFFFF !important;
    text-shadow: 0 0 6px rgba(255,255,255,0.15);
}

h4, h5, h6 {
    color: #E6E6E6 !important;
}

/* ================== 表单标签 ================== */
label, .stSelectbox label, .stSlider label, .stTextInput label {
    color: #EAEAEA !important;
    font-weight: 500;
}

/* ================== Radio 文本 ================== */
div[data-baseweb="radio"] label span {
    color: #F0F0F0 !important;
}

/* ================== Expander ================== */
details summary {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ================== Metric ================== */
[data-testid="stMetric"] label {
    color: #B8C7E0 !important;
}

[data-testid="stMetric"] div {
    color: #FFFFFF !important;
}

/* ================== 提示信息 ================== */
.stAlert p {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* 隐藏 Streamlit 顶部白色 Header */
header[data-testid="stHeader"] {
    display: none;
}

/* 去掉页面顶部多余空白 */
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏导航 ====================
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=100)
st.sidebar.title("导航菜单")

page = st.sidebar.radio(
    "选择功能模块",
    ["项目首页", "专业数据分析", "期末成绩预测"],
    index=0
)

# ==================== 页面1：项目首页 ====================
if page == "项目首页":
    # 首页自定义样式
    st.markdown("""
    <style>
    .section {
        padding: 20px 0 10px 0;
        border-bottom: 1px solid #2A2A2A;
    }
    .card {
        background: linear-gradient(145deg, #141922, #0D1117);
        padding: 18px;
        border-radius: 16px;
        box-shadow: inset 0 0 0 1px #1F2933;
        height: 100%;
    }
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    .card-text {
        color: #E0E0E0;
        line-height: 1.7;
        font-size: 15px;
    }
    .tech {
        background-color: #111827;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        color: #EAEAEA;
        box-shadow: inset 0 0 0 1px #1F2933;
    }
    </style>
    """, unsafe_allow_html=True)

    # 页面标题
    st.markdown("## 🎓 学生成绩分析与预测系统")
    st.markdown(
        "<span style='color:#B0B0B0;font-size:16px'>基于 Streamlit + 机器学习的学生成绩智能分析平台</span>",
        unsafe_allow_html=True
    )

    # 项目概述
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("### 📌 项目概述")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        本项目是一个基于 **Streamlit** 的学生成绩分析与预测系统，
        通过 **数据可视化 + 机器学习模型**，帮助教师和学生深入了解学习状态，
        并对期末成绩进行智能预测。
        **主要特点：**
        - 📊 多维度成绩数据可视化分析
        - 🧠 基于机器学习的成绩预测模型
        - 🎯 支持个性化学习行为分析
        - ⚡ 操作简洁，结果直观，适合教学场景
        """)
    with col2:
        st.image(
            "专业数据分析.png",
            use_container_width=True
        )

    # 项目目标
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("### 🚀 项目目标")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 目标一：数据分析</div>
            <div class="card-text">
                • 识别成绩影响因素<br>
                • 探索成绩变化趋势<br>
                • 提供数据支撑决策
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 目标二：可视化展示</div>
            <div class="card-text">
                • 专业对比分析<br>
                • 性别差异研究<br>
                • 学习行为识别
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 目标三：成绩预测</div>
            <div class="card-text">
                • 构建预测模型<br>
                • 个性化成绩预测<br>
                • 提前干预预警
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 技术架构
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("### 🛠 技术架构")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("<div class='tech'>Streamlit<br>前端框架</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("<div class='tech'>Pandas / NumPy<br>数据处理</div>", unsafe_allow_html=True)
    with t3:
        st.markdown("<div class='tech'>Matplotlib / Plotly<br>数据可视化</div>", unsafe_allow_html=True)
    with t4:
        st.markdown("<div class='tech'>Scikit-learn / XGBoost<br>机器学习</div>", unsafe_allow_html=True)

# ==================== 页面2：专业数据分析 ====================
elif page == "专业数据分析":
    st.markdown("# 📈 专业数据分析")

    if df.empty:
        st.warning("暂无数据，请先上传或生成数据")
    else:
        # 辅助函数：将 DataFrame 转为 Matplotlib 表格图片（适配深色主题）
        def df_to_table_image(df, title="", figsize=(5.5, 4.5), dpi=140):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.axis('off')

            if title:
                ax.set_title(title, fontsize=13, color='white', pad=20)

            # 绘制表格
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                rowLabels=df.index if df.index.name is not None else None,
                loc='center',
                cellLoc='center',
                colWidths=[0.32] * len(df.columns)
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.3, 1.6)

            # 深色主题美化
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # 表头
                    cell.set_facecolor('#1f2a44')
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor('#0f1626' if i % 2 == 1 else '#141b2e')
                cell.set_edgecolor('#2a3b5a')
                cell.set_text_props(color='#d0d8e0')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,
                        facecolor='#0E1117', transparent=False)
            buf.seek(0)
            plt.close(fig)
            return buf

        # 图表1 + 右边图片
        cols1 = st.columns([2.2, 1])
        with cols1[0]:
            with st.container(height=520):
                st.subheader("1. 各专业男女性别比例")
                gender_ratio = df.groupby('专业')['性别'].value_counts(normalize=True).unstack(fill_value=0)
                gender_ratio = gender_ratio.reindex(columns=['男', '女'], fill_value=0).sort_values('男', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 5.5))
                x = np.arange(len(gender_ratio))
                width = 0.35
                ax.bar(x - width/2, gender_ratio['男'], width, label='男', color='#4DA9FF', edgecolor='white')
                ax.bar(x + width/2, gender_ratio['女'], width, label='女', color='#FF6B9D', edgecolor='white')
                ax.set_title('各专业男女性别比例（双层柱状图）', fontsize=16, pad=20)
                ax.set_ylabel('占比')
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.set_yticklabels([f'{int(i*100)}%' for i in ax.get_yticks()])
                ax.set_xticks(x)
                ax.set_xticklabels(gender_ratio.index, rotation=30, ha='right')
                ax.legend()

                for bar in ax.patches:
                    h = bar.get_height()
                    if h > 0.02:
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.1%}',
                                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

                for spine in ax.spines.values():
                    spine.set_color('white')

                st.pyplot(fig)
                plt.close(fig)

        with cols1[1]:
            with st.container(height=520):
                with st.expander("1. 各专业性别比例明细表", expanded=True):
                    ratio_table = (gender_ratio * 100).round(1).astype(str) + '%'
                    ratio_table['总人数'] = df['专业'].value_counts().reindex(ratio_table.index)
                    ratio_table = ratio_table[['总人数', '男', '女']]

                    img_buf = df_to_table_image(
                        ratio_table,
                        title="各专业性别比例明细"
                    )
                    st.image(img_buf)

        st.markdown("---")

        # 图表2 + 右边图片
        cols2 = st.columns([2.2, 1])
        with cols2[0]:
            with st.container(height=680):
                st.subheader("2. 各专业学习投入与成绩对比分析")
                major_stats = df.groupby('专业').agg({
                    '每周学习时长（小时）': 'mean',
                    '期中考试分数': 'mean',
                    '期末考试分数': 'mean'
                }).round(2).sort_values('期末考试分数', ascending=False)

                fig, ax1 = plt.subplots(figsize=(12, 7.5))
                bars = ax1.bar(major_stats.index, major_stats['每周学习时长（小时）'],
                               color='#5DADE2', alpha=0.9, label='平均每周学习时长（小时）', width=0.6)
                ax1.set_ylabel('学习时长（小时）', color='#5DADE2', fontsize=13, fontweight='bold')
                ax1.tick_params(axis='y', labelcolor='#5DADE2')
                ax1.set_ylim(0, major_stats['每周学习时长（小时）'].max() * 1.2)

                for bar in bars:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}h',
                             ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

                ax2 = ax1.twinx()
                scores = pd.concat([major_stats['期中考试分数'], major_stats['期末考试分数']])
                padding = (scores.max() - scores.min()) * 0.4
                y_min = max(50, scores.min() - padding)
                y_max = min(100, scores.max() + padding)
                ax2.set_ylim(y_min, y_max)

                ax2.plot(major_stats.index, major_stats['期中考试分数'], 'o-', linewidth=4, color='#00D4B5',
                         label='平均期中成绩', markersize=9)
                ax2.plot(major_stats.index, major_stats['期末考试分数'], 's-', linewidth=4, color='#FFB866',
                         label='平均期末成绩', markersize=9)

                offset = (y_max - y_min) * 0.03
                for i, (mid, final) in enumerate(zip(major_stats['期中考试分数'], major_stats['期末考试分数'])):
                    ax2.text(i, mid + offset, f'{mid}', ha='center', va='bottom',
                             color='#00D4B5', fontsize=10, fontweight='bold')
                    ax2.text(i, final + offset, f'{final}', ha='center', va='bottom',
                             color='#FFB866', fontsize=10, fontweight='bold')

                ax2.set_ylabel('平均成绩（分）', color='white', fontsize=13, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='white')

                ax1.set_title('各专业平均学习时间与平均成绩对比\n（柱状图=学习时长 | 折线图=期中/期末成绩）',
                              fontsize=17, pad=40, color='white', fontweight='bold')
                plt.xticks(rotation=30, ha='right')

                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(handles1 + handles2, labels1 + labels2,
                           loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=12)
                plt.subplots_adjust(bottom=0.25, top=0.85)

                for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
                    spine.set_color('white')

                st.pyplot(fig)
                plt.close(fig)

        with cols2[1]:
            with st.container(height=680):
                with st.expander("2. 各专业学习投入与成绩明细", expanded=True):
                    img_buf = df_to_table_image(
                        major_stats,
                        title="各专业学习投入与成绩明细"
                    )
                    st.image(img_buf)

        st.markdown("---")

        # 图表3 + 右边图片
        cols3 = st.columns([2.2, 1])
        with cols3[0]:
            with st.container(height=560):
                st.subheader("3. 各专业平均上课出勤率分析")
                attendance = df.groupby('专业')['上课出勤率'].mean().sort_values(ascending=False)
                colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(attendance)))
                fig, ax = plt.subplots(figsize=(11, 6.5))
                bars = ax.bar(attendance.index, attendance.values, color=colors, edgecolor='white',
                              linewidth=1.2, width=0.7)
                ax.set_title('各专业平均上课出勤率\n（颜色越深 = 出勤率越高）',
                             fontsize=18, pad=30, color='white', fontweight='bold')
                ax.set_ylabel('平均出勤率', color='white', fontsize=13)
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.1))
                ax.set_yticklabels([f'{int(x*100)}%' for x in ax.get_yticks()])

                for bar, value in zip(bars, attendance.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                            f'{value:.2%}', ha='center', va='bottom',
                            color='white', fontsize=11, fontweight='bold')

                plt.xticks(rotation=30, ha='right')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.3)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                           norm=plt.Normalize(vmin=attendance.min(), vmax=attendance.max()))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cax)
                cbar.set_label('出勤率高 → 低', color='white', fontsize=12)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                for spine in ax.spines.values():
                    spine.set_color('white')

                st.pyplot(fig)
                plt.close(fig)

        with cols3[1]:
            with st.container(height=560):
                with st.expander("3. 各专业平均出勤率明细", expanded=True):
                    attendance_df = pd.DataFrame({
                        '平均出勤率': attendance.round(4),
                        '平均出勤率(%)': (attendance * 100).round(2).astype(str) + '%'
                    })

                    img_buf = df_to_table_image(
                        attendance_df,
                        title="各专业平均出勤率明细"
                    )
                    st.image(img_buf)

        st.markdown("---")

        # 图表4：大数据管理专业核心指标（4个柱形图横向排列：1×4布局）
        with st.container(height=380):  # 调整容器高度适配横向布局
            st.subheader("4. 大数据管理专业核心指标")

            bd = df[df['专业'] == '大数据管理'] if '大数据管理' in df['专业'].unique() else df.head(50)

            if not bd.empty:
                # 计算四个指标
                avg_attendance = bd['上课出勤率'].mean()
                avg_final_score = bd['期末考试分数'].mean()
                pass_rate = (bd['期末考试分数'] >= 60).mean()
                avg_study_hours = bd['每周学习时长（小时）'].mean()

                # 创建 1×4 横向子图布局（宽度更宽，高度更窄）
                fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))  # 1行4列，宽18高5.5适配横向显示
                fig.suptitle("大数据管理专业关键指标（横向排列）", fontsize=16, color='white', y=0.95)

                # 调整子图间距（横向间距wspace=0.4，纵向间距hspace=0.2）
                plt.subplots_adjust(hspace=0.2, wspace=0.4, top=0.85, bottom=0.15)

                # 子图1: 平均出勤率（横向排列第1个）
                ax1 = axes[0]
                ax1.bar(['平均出勤率'], [avg_attendance * 100], color='#2ECC71', width=0.5)  # 调整柱形宽度
                ax1.set_ylim(0, 110)
                ax1.set_ylabel('百分比 (%)', color='white', fontsize=10)  # 缩小标签字体
                ax1.set_title('平均出勤率', color='white', fontsize=12, pad=8)  # 调整标题大小和间距
                ax1.text(0, avg_attendance * 100 + 3, f'{avg_attendance:.1%}',
                         ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
                ax1.tick_params(colors='white', labelsize=9)
                ax1.set_xticks([])
                for spine in ax1.spines.values():
                    spine.set_color('white')

                # 子图2: 平均期末成绩（横向排列第2个）
                ax2 = axes[1]
                ax2.bar(['平均期末成绩'], [avg_final_score], color='#3498DB', width=0.5)
                ax2.set_ylim(0, 110)
                ax2.set_ylabel('分数', color='white', fontsize=10)
                ax2.set_title('平均期末成绩', color='white', fontsize=12, pad=8)
                ax2.text(0, avg_final_score + 2, f'{avg_final_score:.1f}',
                         ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
                ax2.tick_params(colors='white', labelsize=9)
                ax2.set_xticks([])
                for spine in ax2.spines.values():
                    spine.set_color('white')

                # 子图3: 及格率（横向排列第3个）
                ax3 = axes[2]
                ax3.bar(['及格率'], [pass_rate * 100], color='#E74C3C', width=0.5)
                ax3.set_ylim(0, 110)
                ax3.set_ylabel('百分比 (%)', color='white', fontsize=10)
                ax3.set_title('及格率', color='white', fontsize=12, pad=8)
                ax3.text(0, pass_rate * 100 + 3, f'{pass_rate:.1%}',
                         ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
                ax3.tick_params(colors='white', labelsize=9)
                ax3.set_xticks([])
                for spine in ax3.spines.values():
                    spine.set_color('white')

                # 子图4: 平均学习时长（横向排列第4个）
                ax4 = axes[3]
                ax4.bar(['平均学习时长'], [avg_study_hours], color='#F39C12', width=0.5)
                ax4.set_ylim(0, max(avg_study_hours * 1.4, 40))  # 动态上限
                ax4.set_ylabel('小时/周', color='white', fontsize=10)
                ax4.set_title('平均学习时长', color='white', fontsize=12, pad=8)
                ax4.text(0, avg_study_hours + 1.2, f'{avg_study_hours:.1f}h',
                         ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
                ax4.tick_params(colors='white', labelsize=9)
                ax4.set_xticks([])
                for spine in ax4.spines.values():
                    spine.set_color('white')

                # 整体背景与边框（保持深色主题一致）
                fig.patch.set_facecolor('#0E1117')
                for ax in axes.flat:
                    ax.set_facecolor('#0E1117')

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.info("暂无大数据管理专业数据")

# ==================== 页面3：期末成绩预测 ====================
elif page == "期末成绩预测":
    st.title("🔮 期末成绩预测")
    if model is None or df.empty:
        st.error("❌ 预测功能不可用：模型或数据文件缺失")
    else:
        st.markdown("### 填写以下信息，立即获取精准预测结果")
        with st.form("预测表单"):
            col1, col2 = st.columns(2)
            with col1:
                student_id = st.text_input("学号（仅展示用）", "2023123456")
                gender = st.selectbox("性别", ["男", "女"])
                major = st.selectbox("专业", sorted(df['专业'].unique()))
                study_hours = st.slider("每周学习时长（小时）", 5, 40, 20)

            with col2:
                attendance = st.slider("上课出勤率", 0.60, 1.00, 0.90, step=0.01)
                midterm = st.slider("期中考试分数", 0, 100, 75)
                homework = st.slider("作业完成率", 0.60, 1.00, 0.90, step=0.01)

            submitted = st.form_submit_button(
                "立即预测期末成绩",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                try:
                    # 转换类别特征
                    g_code = le_gender.transform([gender])[0]
                    m_code = le_major.transform([major])[0]

                    # 构造输入数据
                    input_data = np.array([[
                        g_code, m_code, study_hours,
                        attendance, midterm, homework
                    ]])

                    # 预测成绩
                    pred = model.predict(input_data)[0]
                    st.markdown(f"## 预测期末成绩：**{pred:.2f} 分**")

                    # 结果提示
                    if pred >= 60:
                        st.balloons()
                        st.success("恭喜！极大概率及格！")
                        st.image("https://thumbs.dreamstime.com/b/group-business-people-meeting-18988469.jpg")
                    else:
                        st.error("有挂科风险！请引起重视")
                        st.image("https://images.unsplash.com/photo-1542744095-291d1f67b221?w=800")

                except Exception as e:
                    st.error(f"预测失败：{str(e)}")

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        学生成绩分析与预测系统 · 模型分离 · 高精度预测 · 黑底高颜值完整版
    </div>
""", unsafe_allow_html=True)
