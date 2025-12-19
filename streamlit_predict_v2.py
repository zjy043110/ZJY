# 第8章/streamlit_predict_v2.PY
import streamlit as st
import pickle
import pandas as pd

# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="企鹅分类器",  # 页面标题
    page_icon=":penguin:",  # 页面图标
    layout="wide",
)

# 使用侧边栏实现多页面显示效果
with st.sidebar:
    st.image('images/right_logo.png', width=100)
    st.title("请选择页面")
    page = st.selectbox("请选择页面", ["简介页面", "预测分类页面"], label_visibility="collapsed")

if page == "简介页面":
    st.title("企鹅分类器:penguin:")
    st.header("数据集介绍")
    st.markdown("""**帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，也可以作为机器学习入门练习。**
该数据集是由Gorman等收集，并发布在一个名为palmerpenguins的R语言包，以对南极企鹅种类进行分类和研究。
该数据集记录了344行观测数据，包含3个不同物种的企鹅：阿德利企鹅、巴布亚企鹅和帽带企鹅的各种信息。""")
    st.header("三种企鹅的卡通图像:")
    st.image('images/penguins.png')

elif page == "预测分类页面":
    st.header("预测企鹅分类")
    st.markdown("这个Web应用是基于帕尔默群岛企鹅数据集构建的模型。只需输入6个信息，就可以预测企鹅的物种，使用下面的表单开始预测吧！")

    # 该页面是3:1:2的列布局
    col_form, col1, col_logo = st.columns([3, 1, 2])
    with col_form:
        # 运用表单和表单提交按钮
        with st.form('user_inputs'):
            island = st.selectbox('企鹅栖息的岛屿', options=['托森岛', '比斯科岛', '德里岛'])
            sex = st.selectbox('性别', options=['雄性', '雌性'])
            
            bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)
            bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)
            flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)
            body_mass = st.number_input('身体质量（克）', min_value=0.0)
            submitted = st.form_submit_button("预测分类")

    # 初始化数据预处理格式中与岛屿相关的变量
    island_dream, island_torgerson, island_biscoe = 0, 0, 0
    # 根据用户输入的岛屿数据更改对应的值
    if island == '比斯科岛':
        island_biscoe = 1
    elif island == '德里岛':
        island_dream = 1
    elif island == '托森岛':
        island_torgerson = 1

    # 初始化数据预处理格式中与性别相关的变量
    sex_female, sex_male = 0, 0
    # 根据用户输入的性别数据更改对应的值
    if sex == '雌性':
        sex_female = 1
    elif sex == '雄性':
        sex_male = 1

    format_data = [bill_length, bill_depth, flipper_length, body_mass,
                   island_dream, island_torgerson, island_biscoe, sex_male, sex_female]

    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象
    with open('rfc_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)

    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的映射对象
    with open('output_uniques.pkl', 'rb') as f:
        output_uniques_map = pickle.load(f)

    if submitted:
        format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)
        # 使用模型对格式化后的数据format_data进行预测，返回预测的类别代码
        predict_result_code = rfc_model.predict(format_data_df)
        # 将类别代码映射到具体的类别名称
        predict_result_species = output_uniques_map[predict_result_code[0]]
        st.write(f"根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**")

    with col_logo:
        if not submitted:
            st.image('images/right_logo.png', width=300)
        else:
            st.image(f'images/{predict_result_species}.png', width=300)
