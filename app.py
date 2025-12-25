# 导入Streamlit库，用于快速构建Web应用
import streamlit as st
# 导入Pandas库，用于数据处理和分析
import pandas as pd
# 导入NumPy库，用于数值计算
import numpy as np
# 导入Matplotlib的pyplot模块，用于绘制图表
import matplotlib.pyplot as plt
# 导入io模块，用于处理字节流
import io
# 导入joblib库，用于加载预训练的机器学习模型文件
import joblib
# 导入warnings模块，用于控制警告信息的显示
import warnings
# 从matplotlib的工具模块导入坐标轴定位器，用于更精细的图表布局控制
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==================== 全局配置与初始化 ====================
# 忽略Matplotlib库产生的UserWarning类型警告，避免控制台输出冗余信息
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置Matplotlib支持中文显示，指定可用的中文字体列表
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# 解决Matplotlib中负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 批量更新Matplotlib的绘图样式参数，适配深色主题
plt.rcParams.update({
    'figure.facecolor': '#0E1117',        # 图表整体背景色设为深灰黑色
    'axes.facecolor': '#0E1117',          # 坐标轴区域背景色设为深灰黑色
    'text.color': 'black',                # 所有文字颜色设为黑色（适配浅色背景）
    'axes.labelcolor': 'black',           # 坐标轴标签颜色设为黑色
    'xtick.color': 'black',               # x轴刻度文字颜色设为黑色
    'ytick.color': 'black',               # y轴刻度文字颜色设为黑色
    'axes.edgecolor': 'black',            # 坐标轴边框颜色设为黑色
    'grid.color': '#404040',              # 网格线颜色设为浅灰色
    'grid.alpha': 0.4,                    # 网格线透明度设为0.4（半透明）
    # 调整图表子图的边距，消除默认空白，保证不同图表高度一致
    'figure.subplot.left': 0.08,          # 子图左侧边距
    'figure.subplot.right': 0.92,         # 子图右侧边距
    'figure.subplot.top': 0.95,           # 子图顶部边距
    'figure.subplot.bottom': 0.08,        # 子图底部边距
})

# 配置Streamlit页面的基础属性
st.set_page_config(
    page_title="学生成绩分析与预测系统",  # 网页标题
    layout="wide",                       # 页面布局设为宽屏模式
    page_icon="🎓",                      # 网页标签图标
    initial_sidebar_state="expanded"     # 初始状态下侧边栏展开
)

# ==================== 全局常量（精准控制高度）====================
UNIFIED_HEIGHT = 5.0  # 定义所有图表/表格的统一高度（单位：英寸）
COLUMN_RATIO = [2.5, 1]  # 定义页面列的宽度比例为2.5:1（左图:right表）
DPI = 100  # 定义统一的图片分辨率（每英寸像素数），保证像素级高度匹配
LEFT_FIG_WIDTH = 12.0  # 定义左侧图表的宽度（单位：英寸）
RIGHT_FIG_WIDTH = 8.0  # 定义右侧表格图片的宽度（单位：英寸）

# ==================== 缓存函数：数据与模型加载 ====================
# 使用st.cache_resource装饰器缓存模型加载结果，避免重复加载提升性能
@st.cache_resource
def load_model():
    """加载训练好的预测模型和标签编码器"""
    try:
        # 从指定路径加载预训练的成绩预测模型
        model = joblib.load("models/xgb_final_predictor.pkl")
        # 加载性别标签编码器（用于将性别字符串转为模型可识别的数值）
        le_gender = joblib.load("models/le_gender.pkl")
        # 加载专业标签编码器（用于将专业字符串转为模型可识别的数值）
        le_major = joblib.load("models/le_major.pkl")
        # 返回加载的模型和编码器
        return model, le_gender, le_major
    except Exception as e:
        # 加载失败时在页面显示警告信息
        st.warning(f"模型加载失败：{str(e)}")
        # 加载失败返回None
        return None, None, None

# 使用st.cache_data装饰器缓存数据加载结果，避免重复生成模拟数据
@st.cache_data
def load_data():
    """加载学生数据，文件不存在时生成模拟数据"""
    try:
        # 尝试从CSV文件读取真实学生数据
        return pd.read_csv("student_data_adjusted_rounded.csv")
    except Exception:
        # 读取失败时显示警告，告知用户使用模拟数据
        st.warning("数据文件未找到，使用模拟数据")
        # 定义模拟数据的专业列表
        majors = ['大数据管理', '人工智能', '计算机科学', '软件工程', '信息安全']
        # 定义模拟数据的性别列表
        genders = ['男', '女']
        # 构建模拟数据集字典
        data = {
            '专业': np.random.choice(majors, 200),  # 随机生成200条专业数据
            '性别': np.random.choice(genders, 200),  # 随机生成200条性别数据
            '每周学习时长（小时）': np.random.uniform(5, 40, 200),  # 生成5-40小时的学习时长
            '上课出勤率': np.random.uniform(0.6, 1.0, 200),  # 生成0.6-1.0的出勤率
            '期中考试分数': np.random.uniform(50, 95, 200),  # 生成50-95分的期中成绩
            '期末考试分数': np.random.uniform(50, 95, 200),  # 生成50-95分的期末成绩
            '作业完成率': np.random.uniform(0.6, 1.0, 200),  # 生成0.6-1.0的作业完成率
            # 生成200个学号（2023开头，6位数字补零）
            '学号': [f"2023{str(i).zfill(6)}" for i in range(1, 201)]
        }
        # 将字典转换为Pandas DataFrame并返回
        return pd.DataFrame(data)

# ==================== 辅助函数 ====================
def df_to_table_image(df, title=""):
    """生成高度统一的表格图片（纯Matplotlib实现）"""
    # 创建指定尺寸的绘图画布（宽度适配右侧，高度全局统一）
    fig, ax = plt.subplots(
        figsize=(RIGHT_FIG_WIDTH, UNIFIED_HEIGHT),  # 画布尺寸（宽，高）
        dpi=DPI,                                     # 画布分辨率
        frameon=False                                # 关闭画布边框，减少额外高度
    )
    # 关闭坐标轴显示（表格不需要坐标轴）
    ax.axis('off')
    
    # 如果传入了标题，则绘制表格标题
    if title:
        ax.text(
            0.5, 0.98, title,                    # 标题文字内容和位置（居中靠上）
            ha='center', va='top',               # 水平居中，垂直靠上对齐
            fontsize=14, color='black',          # 字体大小14，颜色黑色
            transform=ax.transAxes,              # 使用相对坐标系（0-1）
            # 标题背景框样式：圆角，内边距0.2，背景色深灰，透明度0.8
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f2a44', alpha=0.8)
        )
    
    # 在画布上绘制表格（核心代码）
    table = ax.table(
        cellText=df.values,                      # 表格数据（DataFrame的值）
        colLabels=df.columns,                    # 表格列标题（DataFrame的列名）
        loc='center',                            # 表格位置居中
        cellLoc='center',                        # 单元格内容居中对齐
        # 列宽度自适应：每列宽度=1/列数
        colWidths=[1/len(df.columns)] * len(df.columns),
        # 精准控制表格位置和尺寸，占满5英寸高度（[左, 下, 宽, 高]）
        bbox=[0.02, 0.02, 0.96, 0.90]
    )
    
    # 表格样式优化（放大且占满高度）
    table.auto_set_font_size(False)  # 关闭自动字体大小调整，手动设置
    table.set_fontsize(11)           # 设置表格字体大小为11号
    table.scale(1.0, 2.0)            # 表格缩放：水平1倍，垂直2倍（填满高度）

    # 遍历所有单元格，设置样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行（第0行）
            cell.set_facecolor('#1f2a44')  # 表头背景色设为深灰蓝
            # 表头文字：白色、加粗、12号字体
            cell.set_text_props(color='white', weight='bold', fontsize=12)
        else:  # 内容行
            # 内容行背景色交替（深灰/稍浅灰），提升可读性
            cell.set_facecolor('#0f1626' if i % 2 == 1 else '#141b2e')
            # 内容文字：浅灰蓝、11号字体
            cell.set_text_props(color='#d0d8e0', fontsize=11)
        cell.set_edgecolor('#2a3b5a')  # 单元格边框色设为灰蓝色
        # 强制设置单元格高度，保证不同行数的表格高度统一
        cell.set_height(1/(len(df)+1.5))

    # 创建字节流缓冲区，用于保存图片（无需写入本地文件）
    buf = io.BytesIO()
    # 将表格保存为PNG图片到缓冲区
    plt.savefig(
        buf,                        # 保存到字节流缓冲区
        format='png',               # 图片格式为PNG
        dpi=DPI,                    # 图片分辨率
        bbox_inches='tight',        # 紧凑布局，去除额外空白
        pad_inches=0.0,             # 完全去掉内边距，保证高度精准
        facecolor='#0E1117'         # 图片背景色设为深灰黑色
    )
    # 将缓冲区指针移到开头，准备读取
    buf.seek(0)
    # 关闭Matplotlib画布，释放内存
    plt.close(fig)
    # 返回包含图片数据的字节流
    return buf

# ==================== 页面内容函数 ====================
def show_home_page():
    """展示项目首页内容"""
    # 设置页面主标题
    st.title("🎓 学生成绩分析与预测系统")
    # 设置页面副标题/说明文字
    st.caption("基于 Streamlit + 机器学习的学生成绩智能分析平台")

    # 设置二级标题：项目概述
    st.subheader("📌 项目概述")
    # 创建3:2比例的两列布局
    col1, col2 = st.columns([3, 2])
    with col1:  # 左侧列
        # 显示项目概述的Markdown文本
        st.markdown("""
        本项目通过数据可视化与机器学习模型，帮助教师和学生：
        - 多维度分析成绩影响因素
        - 探索不同专业的学习表现差异
        - 对期末成绩进行智能预测
        - 支持个性化学习行为洞察
        """)
    with col2:  # 右侧列
        try:
            # 尝试加载并显示项目相关图片，宽度适配列宽
            st.image("专业数据分析.png", use_container_width=True)
        except:
            # 图片加载失败时显示提示信息
            st.info("示例图片未找到")

    # 设置二级标题：项目目标
    st.subheader("🚀 项目目标")
    # 创建三等分列布局
    cols = st.columns(3)
    with cols[0]:  # 第一列
        # 显示数据分析目标的信息卡片
        st.info("**数据分析**\n识别关键影响因素\n发现成绩变化趋势")
    with cols[1]:  # 第二列
        # 显示可视化展示目标的信息卡片
        st.info("**可视化展示**\n专业对比\n性别差异\n学习行为分析")
    with cols[2]:  # 第三列
        # 显示成绩预测目标的信息卡片
        st.info("**成绩预测**\nXGBoost 模型\n个性化预测\n提前预警")

    # 设置二级标题：技术栈
    st.subheader("🛠 技术栈")
    # 创建四等分列布局
    cols = st.columns(4)
    with cols[0]:  # 第一列
        # 显示Streamlit技术说明
        st.markdown("**Streamlit**\n交互界面")
    with cols[1]:  # 第二列
        # 显示Pandas/NumPy技术说明
        st.markdown("**Pandas / NumPy**\n数据处理")
    with cols[2]:  # 第三列
        # 显示Matplotlib技术说明
        st.markdown("**Matplotlib**\n图表绘制")
    with cols[3]:  # 第四列
        # 显示XGBoost技术说明
        st.markdown("**XGBoost**\n机器学习")

def show_major_analysis_page(df):
    """展示专业数据分析页面"""
    # 设置页面主标题
    st.title("📈 专业数据分析")

    # 检查数据是否为空，为空则显示警告并返回
    if df.empty:
        st.warning("暂无可用数据")
        return

    # ==================== 模块1：各专业男女性别比例 ====================
    # 设置二级标题：各专业男女性别比例
    st.subheader("1. 各专业男女性别比例")
    # 创建2.5:1比例的两列布局
    col1, col2 = st.columns(COLUMN_RATIO)
    
    with col1:  # 左侧列：绘制性别比例柱状图
        # 按专业分组，统计性别数量并归一化（计算比例），缺失值填充0
        gender_ratio = df.groupby('专业')['性别'].value_counts(normalize=True).unstack(fill_value=0)
        # 重新索引列，确保只有"男"和"女"两列，缺失列填充0
        gender_ratio = gender_ratio.reindex(columns=['男', '女'], fill_value=0)\
                                  .sort_values('男', ascending=False)  # 按男性比例降序排序

        # 创建指定尺寸的绘图画布（左侧宽度，统一高度）
        fig, ax = plt.subplots(
            figsize=(LEFT_FIG_WIDTH, UNIFIED_HEIGHT),  # 画布尺寸
            dpi=DPI,                                   # 分辨率
            frameon=False                              # 关闭边框
        )
        
        # 生成x轴坐标（专业数量个点）
        x = np.arange(len(gender_ratio))
        width = 0.35  # 柱子宽度
        # 绘制男性比例柱状图（蓝色）
        ax.bar(x - width/2, gender_ratio['男'], width, label='男', color='#4DA9FF')
        # 绘制女性比例柱状图（粉色）
        ax.bar(x + width/2, gender_ratio['女'], width, label='女', color='#FF6B9D')
        
        # 优化图表显示（保证清晰，不改变高度）
        ax.set_ylabel('占比', fontsize=10)  # 设置y轴标签，字体10号
        ax.set_ylim(0, 1)  # y轴范围设为0-1（比例）
        # 设置y轴刻度标签，字体9号
        ax.set_yticklabels([f'{int(i*100)}%' for i in ax.get_yticks()], fontsize=9)
        ax.set_xticks(x)  # 设置x轴刻度位置
        # 设置x轴刻度标签（专业名），旋转25度，右对齐，字体9号
        ax.set_xticklabels(gender_ratio.index, rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='upper right')  # 设置图例，字体9号，位置右上
        
        # 为每个柱子添加数值标签（百分比）
        for bar in ax.patches:
            h = bar.get_height()  # 获取柱子高度（比例）
            if h > 0.02:  # 只显示比例>2%的标签，避免重叠
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, 
                        f'{h:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 紧凑布局，去除额外空白，保证高度精准
        plt.tight_layout(pad=0.0)
        # 在Streamlit中显示图表，宽度适配列宽
        st.pyplot(fig, use_container_width=True)
        # 关闭画布，释放内存
        plt.close(fig)

    with col2:  # 右侧列：显示性别比例表格图片
        # 将比例转换为百分比，保留1位小数，转为字符串并加%
        ratio_table = (gender_ratio * 100).round(1).astype(str) + '%'
        # 添加总人数列：按专业统计人数
        ratio_table['总人数'] = df['专业'].value_counts().reindex(ratio_table.index)
        # 只保留总人数、男、女三列
        ratio_table = ratio_table[['总人数', '男', '女']]
        # 生成表格图片
        table_img = df_to_table_image(ratio_table, "性别比例明细")
        # 显示表格图片，宽度适配列宽
        st.image(table_img, use_container_width=True)

    # 添加水平分隔线，区分不同模块
    st.divider()

    # ==================== 模块2：各专业学习投入与成绩对比 ====================
    # 设置二级标题：各专业学习投入与成绩对比分析
    st.subheader("2. 各专业学习投入与成绩对比分析")
    # 创建2.5:1比例的两列布局
    col1, col2 = st.columns(COLUMN_RATIO)
    
    with col1:  # 左侧列：绘制学习时长与成绩对比图
        # 按专业分组，计算学习时长、期中成绩、期末成绩的平均值，保留2位小数
        major_stats = df.groupby('专业').agg({
            '每周学习时长（小时）': 'mean',
            '期中考试分数': 'mean',
            '期末考试分数': 'mean'
        }).round(2).sort_values('期末考试分数', ascending=False)  # 按期末成绩降序排序

        # 创建指定尺寸的绘图画布
        fig, ax1 = plt.subplots(
            figsize=(LEFT_FIG_WIDTH, UNIFIED_HEIGHT),
            dpi=DPI,
            frameon=False
        )
        
        # 绘制学习时长柱状图（浅蓝色）
        bars = ax1.bar(major_stats.index, major_stats['每周学习时长（小时）'],
                       color='#5DADE2', alpha=0.9, label='平均每周学习时长', width=0.6)
        # 设置左y轴标签，颜色与柱子一致，字体10号
        ax1.set_ylabel('学习时长（小时）', color='#5DADE2', fontsize=10)
        # 设置左y轴刻度颜色与标签一致，字体9号
        ax1.tick_params(axis='y', labelcolor='#5DADE2', labelsize=9)
        # 设置左y轴范围，顶部留20%空间显示标签
        ax1.set_ylim(0, major_stats['每周学习时长（小时）'].max() * 1.2)

        # 为学习时长柱子添加数值标签
        for bar in bars:
            h = bar.get_height()  # 获取柱子高度
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5, 
                    f'{h:.1f}h', ha='center', va='bottom', fontsize=8)

        # 创建共享x轴的右y轴（双轴图）
        ax2 = ax1.twinx()
        # 绘制期中成绩折线图（青绿色，圆形标记）
        ax2.plot(major_stats.index, major_stats['期中考试分数'], 'o-', 
                color='#00D4B5', label='平均期中成绩', linewidth=2, markersize=4)
        # 绘制期末成绩折线图（橙粉色，方形标记）
        ax2.plot(major_stats.index, major_stats['期末考试分数'], 's-', 
                color='#FF6B9D', label='平均期末成绩', linewidth=2, markersize=4)
        # 设置右y轴标签，字体10号
        ax2.set_ylabel('平均成绩（分）', fontsize=10)
        # 设置右y轴刻度字体9号
        ax2.tick_params(labelsize=9)

        # 优化标签显示（不超出高度）
        # 设置x轴刻度标签，旋转25度，右对齐，字体9号
        ax1.set_xticklabels(major_stats.index, rotation=25, ha='right', fontsize=9)
        # 设置图表标题，字体11号，内边距5
        ax1.set_title('学习时间与成绩对比', fontsize=11, pad=5)
        # 设置图例，位置居中靠下，3列显示，字体9号
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)
        
        # 紧凑布局，去除额外空白
        plt.tight_layout(pad=0.0)
        # 在Streamlit中显示图表
        st.pyplot(fig, use_container_width=True)
        # 关闭画布
        plt.close(fig)

    with col2:  # 右侧列：显示学习投入与成绩表格图片
        # 生成表格图片
        table_img = df_to_table_image(major_stats, "学习投入与成绩明细")
        # 显示表格图片
        st.image(table_img, use_container_width=True)

    # 添加水平分隔线
    st.divider()

    # ==================== 模块3：各专业平均上课出勤率 ====================
    # 设置二级标题：各专业平均上课出勤率
    st.subheader("3. 各专业平均上课出勤率")
    # 创建2.5:1比例的两列布局
    col1, col2 = st.columns(COLUMN_RATIO)
    
    with col1:  # 左侧列：绘制出勤率柱状图
        # 按专业分组，计算平均出勤率，按出勤率降序排序
        attendance = df.groupby('专业')['上课出勤率'].mean().sort_values(ascending=False)
        
        # 创建指定尺寸的绘图画布
        fig, ax = plt.subplots(
            figsize=(LEFT_FIG_WIDTH, UNIFIED_HEIGHT),
            dpi=DPI,
            frameon=False
        )
        # 生成渐变颜色列表（基于viridis配色方案）
        colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(attendance)))
        
        # 绘制出勤率柱状图，使用渐变颜色，宽度0.6
        bars = ax.bar(attendance.index, attendance.values, color=colors, width=0.6)
        # 设置y轴标签，字体10号
        ax.set_ylabel('平均出勤率', fontsize=10)
        # 设置y轴范围0-1
        ax.set_ylim(0, 1)
        # 设置y轴刻度标签，字体9号
        ax.set_yticklabels([f'{int(x*100)}%' for x in ax.get_yticks()], fontsize=9)
        # 设置x轴刻度标签，旋转25度，右对齐，字体9号
        ax.set_xticklabels(attendance.index, rotation=25, ha='right', fontsize=9)
        
        # 为每个柱子添加出勤率数值标签
        for bar, value in zip(bars, attendance.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2%}', ha='center', va='bottom', fontsize=8)
        
        # 设置图表标题，字体11号，内边距5
        ax.set_title('各专业平均出勤率', fontsize=11, pad=5)
        # 紧凑布局
        plt.tight_layout(pad=0.0)
        # 显示图表
        st.pyplot(fig, use_container_width=True)
        # 关闭画布
        plt.close(fig)

    with col2:  # 右侧列：显示出勤率表格图片
        # 构建出勤率DataFrame：原始值（保留4位小数）和百分比字符串
        attendance_df = pd.DataFrame({
            '平均出勤率': attendance.round(4),
            '百分比': (attendance * 100).round(2).astype(str) + '%'
        })
        # 生成表格图片
        table_img = df_to_table_image(attendance_df, "出勤率明细")
        # 显示表格图片
        st.image(table_img, use_container_width=True)

    # 添加水平分隔线
    st.divider()

    # ==================== 模块4：大数据管理专业核心指标 ====================
    # 设置二级标题：大数据管理专业核心指标
    st.subheader("4. 大数据管理专业核心指标")
    # 筛选出大数据管理专业的数据
    bd_df = df[df['专业'] == '大数据管理']
    
    # 检查大数据管理专业是否有数据
    if not bd_df.empty:
        # 计算核心指标
        avg_attendance = bd_df['上课出勤率'].mean()          # 平均出勤率
        avg_final_score = bd_df['期末考试分数'].mean()       # 平均期末成绩
        pass_rate = (bd_df['期末考试分数'] >= 60).mean()     # 及格率（分数≥60的比例）
        avg_study_hours = bd_df['每周学习时长（小时）'].mean()  # 平均学习时长

        # 指标卡片（紧凑布局）：创建四等分列布局
        cols = st.columns(4)
        cols[0].metric("平均出勤率", f"{avg_attendance:.1%}")          # 显示平均出勤率指标卡
        cols[1].metric("平均期末成绩", f"{avg_final_score:.1f} 分")     # 显示平均期末成绩指标卡
        cols[2].metric("及格率", f"{pass_rate:.1%}")                  # 显示及格率指标卡
        cols[3].metric("平均学习时长", f"{avg_study_hours:.1f} 小时/周")  # 显示平均学习时长指标卡

        # 设置三级标题：核心指标对比
        st.markdown("### 核心指标对比")
        # 创建2.5:1比例的两列布局
        col1, col2 = st.columns(COLUMN_RATIO)
        
        with col1:  # 左侧列：绘制核心指标柱状图
            # 创建指定尺寸的绘图画布
            fig, ax = plt.subplots(
                figsize=(LEFT_FIG_WIDTH, UNIFIED_HEIGHT),
                dpi=DPI,
                frameon=False
            )
            # 定义核心指标列表（名称、数值、颜色）
            indicators = [
                ("平均出勤率", avg_attendance * 100, '#2ECC71'),    # 出勤率（转百分比），绿色
                ("平均期末成绩", avg_final_score, '#3498DB'),        # 期末成绩，蓝色
                ("及格率", pass_rate * 100, '#E74C3C'),              # 及格率（转百分比），红色
                ("平均学习时长", avg_study_hours, '#F39C12')         # 学习时长，橙色
            ]

            # 提取指标名称、数值、颜色
            categories = [ind[0] for ind in indicators]
            values = [ind[1] for ind in indicators]
            colors = [ind[2] for ind in indicators]

            # 绘制核心指标柱状图，宽度0.6，白色边框
            bars = ax.bar(categories, values, color=colors, width=0.6, 
                         edgecolor='white', linewidth=1)
            
            # 优化标题和标签
            # 设置图表标题，字体11号，内边距5，白色文字
            ax.set_title('大数据管理专业核心指标', fontsize=11, pad=5, color='white')
            # 设置y轴标签，字体10号，白色文字
            ax.set_ylabel('数值', fontsize=10, color='white')
            # 设置刻度颜色为白色，字体9号
            ax.tick_params(colors='white', labelsize=9)
            # 设置x轴刻度标签，旋转15度，右对齐，字体9号
            ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)

            # 动态设置y轴上限，保证标签显示（顶部留20%空间）
            max_val = max(values)
            ax.set_ylim(0, max_val * 1.2)

            # 为每个柱子添加数值标签（根据指标类型显示不同格式）
            for bar, (label, value, _) in zip(bars, indicators):
                if "出勤率" in label or "及格率" in label:
                    text = f"{value:.1f}%"  # 百分比格式
                elif "成绩" in label:
                    text = f"{value:.1f}分"  # 分数格式
                else:
                    text = f"{value:.1f}h"   # 小时格式
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.02,
                    text,
                    ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold'
                )

            # 深色主题适配
            ax.set_facecolor('#0E1117')  # 坐标轴背景色
            fig.patch.set_facecolor('#0E1117')  # 图表背景色
            # 设置所有坐标轴边框为白色
            for spine in ax.spines.values():
                spine.set_color('white')
            # 添加水平网格线，浅灰色，半透明，虚线
            ax.grid(True, axis='y', alpha=0.25, color='#505050', linestyle='--')
            
            # 紧凑布局
            plt.tight_layout(pad=0.0)
            # 显示图表
            st.pyplot(fig, use_container_width=True)
            # 关闭画布
            plt.close(fig)
        
        with col2:  # 右侧列：显示核心指标表格图片
            # 构建核心指标DataFrame
            bd_stats = pd.DataFrame({
                '指标': ['平均出勤率', '平均期末成绩', '及格率', '平均学习时长'],
                '数值': [
                    f"{avg_attendance:.1%}",
                    f"{avg_final_score:.1f} 分",
                    f"{pass_rate:.1%}",
                    f"{avg_study_hours:.1f} 小时/周"
                ]
            })
            # 生成表格图片
            table_img = df_to_table_image(bd_stats, "大数据管理专业明细")
            # 显示表格图片
            st.image(table_img, use_container_width=True)
    else:
        # 大数据管理专业无数据时显示提示信息
        st.info("暂无 '大数据管理' 专业的数据")

def show_score_prediction_page(model, le_gender, le_major, df):
    """展示期末成绩预测页面"""
    # 设置页面主标题
    st.title("🔮 期末成绩预测")

    # 检查模型是否加载成功
    if model is None:
        # 模型加载失败时显示错误信息
        st.error("预测模型不可用，请检查模型文件是否存在")
        return

    # 设置二级标题：填写以下信息进行预测
    st.subheader("填写以下信息进行预测")
    # 创建表单容器（提交按钮需要放在表单内）
    with st.form("预测表单"):
        # 创建两等分列布局
        col1, col2 = st.columns(2)
        with col1:  # 左侧列：输入基本信息
            # 文本输入框：学号（仅展示，默认值2023123456）
            st.text_input("学号（仅展示）", "2023123456", key="id")
            # 下拉选择框：性别（选项：男、女）
            gender = st.selectbox("性别", ["男", "女"])
            # 下拉选择框：专业（选项为数据中的唯一专业值，排序）
            major = st.selectbox("专业", sorted(df['专业'].unique()))
        with col2:  # 右侧列：输入学习表现信息
            # 滑块：每周学习时长（范围5-40，默认20）
            study_hours = st.slider("每周学习时长（小时）", 5, 40, 20)
            # 滑块：上课出勤率（范围0.60-1.00，步长0.01，默认0.90）
            attendance = st.slider("上课出勤率", 0.60, 1.00, 0.90, step=0.01)
            # 滑块：期中考试分数（范围0-100，默认75）
            midterm = st.slider("期中考试分数", 0, 100, 75)
            # 滑块：作业完成率（范围0.60-1.00，步长0.01，默认0.90）
            homework = st.slider("作业完成率", 0.60, 1.00, 0.90, step=0.01)

        # 创建表单提交按钮，宽度适配容器
        submitted = st.form_submit_button("立即预测", use_container_width=True, type="primary")

        # 当表单提交时执行预测逻辑
        if submitted:
            try:
                # 将性别字符串转换为模型可识别的编码
                g_code = le_gender.transform([gender])[0]
                # 将专业字符串转换为模型可识别的编码
                m_code = le_major.transform([major])[0]
                # 构建模型输入特征数组（二维数组，符合scikit-learn输入格式）
                input_data = np.array([[g_code, m_code, study_hours, attendance, midterm, homework]])
                # 使用模型预测期末成绩
                pred = model.predict(input_data)[0]

                # 显示预测结果（二级标题，加粗显示分数）
                st.subheader(f"预测期末成绩：**{pred:.2f} 分**")
                
                # ==================== 预测结果下方添加图片 ====================
                # 设置图片宽度（可根据需要调整）
                img_width = 400
                
                # 根据分数是否及格显示不同图片
                if pred >= 60:
                    st.success("极大概率及格！继续保持～")  # 显示成功提示
                    st.balloons()  # 触发气球动画效果
                    # 显示及格鼓励的图片（本地图片）
                    st.image(
                        "pass.jpg",  # 及格图片路径
                        caption="🎉 恭喜！继续保持优秀表现～",
                        width=img_width
                    )
                else:
                    st.error("存在挂科风险，建议加强复习与出勤")  # 显示错误提示
                    # 显示不及格提醒的图片（本地图片）
                    st.image(
                        "nopass.jpg",  # 不及格图片路径
                        caption="📖 加油！多投入时间复习，提高出勤率～",
                        width=img_width
                    )
                # ==================== 图片添加结束 ====================
                
            except Exception as e:
                # 预测过程中出错时显示错误信息
                st.error(f"预测出错：{str(e)}")

# ==================== 主程序入口 ====================
def main():
    # 加载机器学习模型和标签编码器
    model, le_gender, le_major = load_model()
    # 加载学生数据（真实数据或模拟数据）
    df = load_data()

    # 创建侧边栏容器
    with st.sidebar:
        # 在侧边栏显示图片（毕业帽图标），宽度100px
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=100)
        # 设置侧边栏标题
        st.title("导航菜单")
        # 侧边栏单选框：选择要显示的页面
        page = st.radio(
            "选择功能模块",
            ["项目首页", "专业数据分析", "期末成绩预测"]
        )

    # 根据用户选择的页面，调用对应的显示函数
    if page == "项目首页":
        show_home_page()
    elif page == "专业数据分析":
        show_major_analysis_page(df)
    elif page == "期末成绩预测":
        show_score_prediction_page(model, le_gender, le_major, df)

    # 添加水平分隔线（页脚上方）
    st.divider()
    # 设置页脚文字
    st.caption("学生成绩分析与预测系统")

# 程序入口：当脚本直接运行时，执行main函数
if __name__ == "__main__":
    main()
