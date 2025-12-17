import streamlit as st
import pandas as pd
import plotly.express as px


def get_dataframe_from_excel() -> pd.DataFrame:
    """
    从Excel文件读取销售数据并进行预处理
    Returns:
        pd.DataFrame: 包含小时数字段的销售数据框
    """
    # 替换为你的Excel文件实际绝对路径
    excel_path = r"D:\streamlit_env\supermarket_sales.xlsx"
    # 读取Excel文件数据
    df = pd.read_excel(
        excel_path,
        sheet_name='销售数据',
        skiprows=1,
        index_col='订单号'
    )
    
    # 从时间列提取小时数，新增小时数字段
    df['小时数'] = pd.to_datetime(df["时间"], format="%H:%M:%S").dt.hour
    
    return df


def add_sidebar_func(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建侧边栏筛选组件，并根据筛选条件返回数据
    核心优化：1. 多选框为空时默认全选 2. 修复query多值匹配语法
    Args:
        df: 原始销售数据框
    Returns:
        pd.DataFrame: 筛选后的销售数据框
    """
    with st.sidebar:
        st.header("请筛选数据：")
        
        # 1. 城市筛选：为空时默认全选
        city_unique = df["城市"].unique()
        city = st.multiselect(
            "请选择城市：",
            options=city_unique,
            default=city_unique,  # 初始全选
        )
        # 空值处理：如果用户清空选择框，重置为全选
        if not city:
            city = city_unique
        
        # 2. 顾客类型筛选：为空时默认全选
        customer_type_unique = df["顾客类型"].unique()
        customer_type = st.multiselect(
            "请选择顾客类型：",
            options=customer_type_unique,
            default=customer_type_unique,  # 初始全选
        )
        # 空值处理：如果用户清空选择框，重置为全选
        if not customer_type:
            customer_type = customer_type_unique
        
        # 3. 性别筛选：为空时默认全选
        gender_unique = df["性别"].unique()
        gender = st.multiselect(
            "请选择性别：",
            options=gender_unique,
            default=gender_unique,  # 初始全选
        )
        # 空值处理：如果用户清空选择框，重置为全选
        if not gender:
            gender = gender_unique
        
        # 关键修复：用 in 替代 == 处理多值匹配
        # 构建筛选条件（in 关键字适配列表类型的多值匹配）
        df_selection = df.query(
            "城市 in @city & 顾客类型 in @customer_type & 性别 in @gender"
        )
    
    return df_selection


def product_line_chart(df: pd.DataFrame) -> px.bar:
    """
    生成按产品类型划分的销售额横向条形图
    Args:
        df: 筛选后的销售数据框
    Returns:
        px.bar: 产品类型销售额图表对象
    """
    # 按产品类型分组计算总销售额并排序（Series.sort_values() 无需 by 参数）
    sales_by_product_line = (
        df.groupby(by=["产品类型"])["总价"]
        .sum()
        .sort_values()  # 移除 by="总价"，直接排序
    )
    
    # 生成横向条形图
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="总价",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>按产品类型划分的销售额</b>",
    )
    
    return fig_product_sales


def hour_chart(df: pd.DataFrame) -> px.bar:
    """
    生成按小时数划分的销售额条形图
    Args:
        df: 筛选后的销售数据框
    Returns:
        px.bar: 小时销售额图表对象
    """
    # 按小时数分组计算总销售额
    sales_by_hour = df.groupby(by=["小时数"])["总价"].sum()
    
    # 生成条形图
    fig_hour_sales = px.bar(
        sales_by_hour,
        x=sales_by_hour.index,
        y="总价",
        title="<b>按小时数划分的销售额</b>",
    )
    
    return fig_hour_sales


def main_page_demo(df: pd.DataFrame) -> None:
    """
    渲染主页面内容，包括关键指标和图表展示
    Args:
        df: 筛选后的销售数据框
    """
    # 页面标题
    st.title(':bar_chart: 销售仪表板')
    
    # 计算关键业务指标（增加空值/异常值处理）
    total_sales = int(df["总价"].sum()) if not df.empty else 0  # 空数据时总销售额为0
    average_rating = round(df["评分"].mean(), 1) if not df.empty and df["评分"].notna().any() else 0.0  # 处理空评分
    
    # 星级展示：先判断是否为有效数字，再转换
    if pd.notna(average_rating) and average_rating > 0:
        star_rating = ":star:" * int(round(average_rating, 0))
    else:
        star_rating = ":star:" * 0  # 无评分时显示0颗星
    
    average_sale_per_transaction = round(df["总价"].mean(), 2) if not df.empty else 0.0  # 空数据时平均销售额为0
    
    # 关键指标展示区
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("总销售额：")
        st.subheader(f"RMB ¥ {total_sales:,}")
    
    with col2:
        st.subheader("顾客平均评分：")
        st.subheader(f"{average_rating} {star_rating}")
    
    with col3:
        st.subheader("每单平均销售额：")
        st.subheader(f"RMB ¥ {average_sale_per_transaction}")
    
    # 分割线
    st.divider()
    
    # 图表展示区（增加空数据判断，避免图表报错）
    if df.empty:
        st.warning("暂无符合筛选条件的数据，请调整筛选条件！")
    else:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            hour_fig = hour_chart(df)
            st.plotly_chart(hour_fig, use_container_width=True)
        
        with chart_col2:
            product_fig = product_line_chart(df)
            st.plotly_chart(product_fig, use_container_width=True)


def run_app() -> None:
    """应用入口函数，初始化并运行整个应用"""
    # 页面基础配置
    st.set_page_config(
        page_title="销售仪表板",
        page_icon=":bar_chart:",
        layout="wide"
    )
    
    # 数据加载与筛选
    sale_df = get_dataframe_from_excel()
    df_selection = add_sidebar_func(sale_df)
    
    # 渲染主页面
    main_page_demo(df_selection)


if __name__ == "__main__":
    run_app()
