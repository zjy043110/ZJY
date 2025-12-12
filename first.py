import streamlit as st
import pandas as pd

# 页面基础配置
st.set_page_config(page_title="学生档案", page_icon="🎀", layout="wide")

st.markdown("""
    <style>
    /* 全局背景与文字 */
    .stApp {
        background: linear-gradient(135deg, #fff5f7 0%, #ffe6ef 50%, #ffd1dc 100%);
        color: #e6398a;
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    /* 主标题样式 */
    h1 {
        color: #d63384;
        text-shadow: 0 0 8px #ffb6c1;
        font-size: 2.5rem;
        letter-spacing: 3px;
        text-align: center;
        border-bottom: 3px dashed #ff99cc;
        padding-bottom: 10px;
    }
    /* 子标题样式 */
    h2, h3 {
        color: #c2185b;
        text-shadow: 0 0 5px #ffc0cb;
        border-left: 5px solid #ff69b4;
        padding-left: 10px;
    }
    /* 指标卡片 */
    .stMetric {
        background: #fff;
        border: 3px solid #ffb6c1;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(255, 182, 193, 0.5);
    }
    .stMetric label {
        color: #d63384 !important;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .stMetric value {
        color: #e91e63 !important;
        font-size: 2.2rem !important;
        font-weight: bold;
    }
    /* 表格样式 */
    .stDataFrame {
        background: #fff;
        border: 3px solid #ff99cc;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(255, 153, 204, 0.4);
    }
    table th {
        background-color: #ff69b4 !important;
        color: #fff !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    table td {
        border: 1px solid #ffc0cb !important;
        color: #c2185b !important;
        background: #fff0f5 !important;
    }
    /* 代码块 */
    .stCode {
        background: #fff0f5 !important;
        border: 3px solid #ffb6c1 !important;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(255, 182, 193, 0.3);
        color: #d63384 !important;
    }
    /* 普通文本 */
    .stText {
        color: #c2185b;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    /* 进度条 */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #ff69b4, #ff99cc, #ffb6c1);
        border-radius: 10px;
        box-shadow: 0 0 8px #ff69b4;
    }
    /* 分割线 */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #ff99cc, transparent);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 1. 主标题（Title）
st.title("🎀 甜甜的学生档案 🍬")
st.markdown("---")

# 2. 基础信息（Header + Text + Markdown）
st.header("📝 可爱小档案")
col1, col2, col3 = st.columns(3)
with col1:
    st.text("✨ 学生昵称: 小甜豆")
    st.markdown("<span style='color:#e91e63;'>▸ 入学时间: 2022.09</span>", unsafe_allow_html=True)
with col2:
    st.text("🎨 喜欢的颜色: 粉色")
    st.markdown("<span style='color:#e91e63;'>▸ 专属徽章: 软萌小码农</span>", unsafe_allow_html=True)
with col3:
    st.text("🍡 小目标: 学会做可爱的可视化")
    st.markdown("<span style='color:#e91e63;'>▸ 心情状态: 超开心😜</span>", unsafe_allow_html=True)
st.markdown("---")

# 3. 技能矩阵（Metric + 进度条）
st.header("💻 编程小技能")
skill_col1, skill_col2, skill_col3 = st.columns(3)
with skill_col1:
    st.metric("Python", "90%", "+4%")
with skill_col2:
    st.metric("SQL", "85%", "+2%")
with skill_col3:
    st.metric("Streamlit", "60%", "+8%")

st.text("🎈 学习进度条")
st.progress(82)  # 模拟进度条
st.markdown("---")

# 4. 任务日志（Table）
st.header("📅 甜甜的任务日志")
task_data = {
    "📆 日期": ["2025.12.11", "2025.12.11", "2025.12.11"],
    "🎯 任务": ["制作可爱档案页", "写甜甜的代码", "做粉色可视化"],
    "🌸 状态": ["✅ 完成啦", "⚪ 努力中", "❣️ 待解锁"],
    "💖 难度": ["★★☆☆☆", "★★★☆☆", "★★☆☆☆"]
}
task_df = pd.DataFrame(task_data)
st.dataframe(task_df, use_container_width=True)  # 表格
st.markdown("---")

# 5. 最新代码成果（Code）
st.header("💌 可爱代码小片段")
code_content = """
# 制作粉色爱心进度条
def cute_progress(rate):
    heart = "❤️" * int(rate * 10)
    empty = "♡" * (10 - int(rate * 10))
    print(f"进度: {heart}{empty} {rate*100}%")

# 调用示例
cute_progress(0.8)  # 进度: ❤️❤️❤️❤️❤️❤️❤️❤️♡♡ 80%
"""
st.code(code_content, language="python")  # 代码块

# 6. 小日记（Markdown + Text）
st.markdown("### 📜 甜甜的小日记")
st.text("▸ 今天学会了做粉色的界面，超开心～")
st.text("▸ 代码写累了就吃一颗草莓糖🍓")
st.text("▸ 下次要做更可爱的可视化！")
st.markdown("<span style='color:#d63384;'>✨ 今日小幸运: 代码一次运行成功～</span>", unsafe_allow_html=True)
