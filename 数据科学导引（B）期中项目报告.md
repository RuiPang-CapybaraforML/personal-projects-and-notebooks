## Part 1: 问题描述和准备工作（代码对应单元组1-2）

在这个项目中，我们使用一个Kaggle上的数据集，对马的生存状况进行预测。数据集包括训练和测试集，都是在原数据基础上，由深度学习生成的。数据集中的特征是马的背景信息和生命体征，由客观测量或主观判断获得。预测的目标变量是三分类的，包括存活(lived)，死亡(died)，安乐死(euthanized)。我们希望探究，能否在训练集上通过EDA、数据处理、建模的过程，研究出一套可行的数据分析模式，从而在测试集上对目标变量做出较为准确的预测。

对于这个三分类问题，直觉上来讲，我们首先考虑的是用决策树模型，因为它和“西瓜书”中判断好瓜坏瓜的问题极其相似。但显然，只用一个决策树模型不能很好地解决我们的问题，因此我们考虑用集成学习来解决这一问题，以提高模型的预测性能和鲁棒性。


### In[1]
我们在这段代码中导入了基本库以及在建模中可能会用到的算法（实际上并没有全部用到，比如K近邻分类、神经网络、高斯过程分类等）

```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import random
import os
from copy import deepcopy
from functools import partial
import gc
import warnings

# 导入模型选择、交叉验证和评估性能的库
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.model_selection import StratifiedKFold, KFold  
    #用于交叉验证，确保数据集的划分有代表性
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.metrics import precision_score, recall_score
    # 性能评估的指标 ROC_AUC Score 准确率 精确率 召回率 对数损失 F1分数
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    # 用于数据预处理，具体为：标准化、归一化，标签编码
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder
    # 用于分类变量的编码，目的是将类别转化为机器学习算法可以处理的数值型类型
from imblearn.under_sampling import RandomUnderSampler # type: ignore
    # 随机下采样，用于处理不平衡的数据，下采样是指减少数据集中某个类别的样本数量，以减少类别之间的不平衡。

import optuna   # 导入超参数调优的库

# 导入梯度提升相关的库
import xgboost as xgb   
import lightgbm as lgb 
    # 用于XGBoost模型和LightGBM模型的实现
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
    # 用于集成学习模型的实现。sklearn.ensemble能提供一系列的机器学习算法
from imblearn.ensemble import BalancedRandomForestClassifier 
    # 用于处理不平衡数据的随机森林模型
from sklearn.impute import KNNImputer
    # 用于使用K近邻方法进行缺失值填充
from sklearn.pipeline import Pipeline
    # 用于构建数据处理和模型训练的流水线
from sklearn.svm import NuSVC, SVC
    # 用于支持向量机模型的实现
from sklearn.neighbors import KNeighborsClassifier
    # 用于K近邻分类模型的实现
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    # 用于逻辑回归模型的实现
from sklearn.neural_network import MLPClassifier
    # 用于多层感知机（神经网络）分类模型的实现
from sklearn.gaussian_process import GaussianProcessClassifier
    # 用于高斯过程分类模型的实现
from sklearn.gaussian_process.kernels import RBF
    # 用于高斯过程分类模型的核函数
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier 
    # 用于Catboost模型的实现
from catboost import Pool 
    # 用于Catboost模型的数据集表示

```


### In[2]

这段代码进行的是seaborn（高级matplotlib）配置

```Python
rc = {
    "axes.facecolor": "#E6FFFF",    # 坐标轴的背景颜色
    "figure.facecolor": "#E6FFFF",     # 整个图表的背景颜色
    "axes.edgecolor": "#000000",     # 坐标轴的边框颜色
    "grid.color": "#EBEBE7",       # 网格线的颜色
    "font.family": "arial",          # 设置字体为arial
    "axes.labelcolor": "#000000",    # 坐标轴标签的颜色
    "xtick.color": "#000000",      # x轴刻度的颜色
    "ytick.color": "#000000",       # y轴刻度的颜色
    "grid.alpha": 0.4            # 网格线透明度
}
sns.set(rc=rc)      # 将上述配置应用到Seaborn的绘图风格中。

# Pandas数据框显示设置
pd.set_option('display.max_columns', None)

# 抑制警告信息
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 定义辅助函数
def print_sl():
    print("=" * 50)
    print()
    # 打印一行五十个等号 用于分割输出结果，增加可读性
def show_na(df, column):
    sns.countplot(x='outcome', data=df[df[column].isnull()])
    plt.show()
    # 绘制一个计数图（countplot），显示在column列中缺失值的行中，outcome列的分布情况。

```


## Part 2: 加载数据（代码对应单元组3）

### In[3]

这段代码主要是用来加载数据，通过对前几行数据的观察，可以帮助我们对数据有一个大致的了解。通过对前几行数据的观察，我们在这里进行了一个相当重要的工作：定义分类变量（categorical_cols）、数值变量（num_cols）和目标变量（target）。显然，对于有些变量，比如surgery，它只有两种映射值（是和否），因此在后面，这类变量的映射值可以转化为离散值进行处理；但对于某些变量，比如rectal_temp，它的映射值全部都是数值。显然，这两类自变量的种类有着较大的差别，在处理时应当分开进行处理。

```Python
train = pd.read_csv('/Users/pangrui/Desktop/train.csv')
test = pd.read_csv('/Users/pangrui/Desktop/test.csv')
sample_submission = pd.read_csv('/Users/pangrui/Desktop/sample_submission.csv')

train_orig = pd.read_csv('/Users/pangrui/Desktop/horse.csv')

# 删除不必要的列
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
    # 删除训练数据集合测试数据集中的id列，因为id列通常不包含对模型有用的信息。
    # drop('id', axis=1, inplace=True): 删除id列，axis=1表示按列删除，inplace=True表示直接在原数据框上进行修改。
print('Data Loaded Succesfully!')
print_sl()
    # 打印“数据加载成功”的提示并调用print_sl()函数（之前定义的辅助函数）打印分隔线，增加可读性。

# 打印数据集的基本信息
print(f'train shape: {train.shape}')
    # 打印训练集的形状（行数和列数）
print(f'are there any null values in train: {train.isnull().any().any()}\n')
    # 打印检查训练数据集中是否存在任何缺失值。
print(f'test shape: {test.shape}')
print(f'are there any null values in test: {test.isnull().any().any()}\n')
    # 同上对测试集进行操作
print(f'train_orig shape: {train_orig.shape}')
print(f'are there any null values in test: {train_orig.isnull().any().any()}\n')
    # 同上对原始训练数据集进行操作

# 定义分类变量、数值变量和目标变量
categorical_cols = ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time',
                   'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces',
                   'abdomen', 'abdomo_appearance', 'surgical_lesion', 'cp_data']

num_cols = ['hospital_number', 'rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein',
           'abdomo_protein', 'lesion_1', 'lesion_2', 'lesion_3']

target = 'outcome'



```

这组代码的运行结果如下：

Data Loaded Succesfully!
    ==================================================

train shape: (1235, 28)
are there any null values in train: True

test shape: (824, 27)
are there any null values in test: True

train_orig shape: (299, 28)
are there any null values in test: True



## Part 3: EDA（代码对应单元组4-8）

实际上，Part 2的加载数据也可以作为EDA的一部分，我们在Part 2中获得了数据集的基本信息，以及是否有缺失值存在，而在这一部分，我们将重点对变量进行统计分析，并将分析结果可视化。主要包含分类变量（categorical_cols）、数值变量（num_cols）和目标变量（target）的分布情况，以及变量之间的相关性分析。

### In[4]

这段代码对目标变量的分布进行了可视化分析，这非常重要，因为实际上，我们对测试数据的预测结果应当和目标变量的分布情况相近，基于这个原理，我们在最后对数据进行了一定的调整，因此对目标变量分布的研究是有必要的。

```Python
def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    
    
    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.2)
    # 设置背景颜色

    s1 = df[col].value_counts()
    N = len(s1)
    # 计算分布情况

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#003366', '#005588', '#0077aa']
    inner_colors = ['#336699', '#5588bb', '#77aabb']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#E6FFFF', linewidth=0)
    ax[0].add_artist(center_circle)
    # 绘制双层饼图，该双层饼图由三部分组成：外层（outer)、内层（inner）和核心（center_circle）

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='Blues', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')
    # 绘制条形图

    f.suptitle(f'{title_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    # 设置图表标题

plot_count(train, 'outcome', 'Target Variable(Outcome) Distribution')
    # 调用函数，绘制图表，展示outcome列的分布情况

```

可视化结果如下：
![示例图片](/原始结果分布.png)
可以看到，安乐死(euthanized)和死亡(died)的数量加起来大致与存活(lived)数量相当，说明这个数据集的*活/死* 样本较为均衡。


### In[5]

对分类变量的分布进行可视化

```Python
plt.figure(figsize=(14, len(categorical_cols)*3))
    # 设置图表大小

for i, col in enumerate(categorical_cols):
    
    plt.subplot(len(categorical_cols)//2 + len(categorical_cols) % 2, 2, i+1)
    # 绘制子图
    sns.countplot(x=col, hue="outcome", data=train, palette='YlOrRd')
    # 绘制条形图
    plt.title(f"{col} countplot by outcome", fontweight = 'bold')
    # 设置子图标题
    plt.ylim(0, train[col].value_counts().max() + 10)
    # 设置y轴范围

    # 遍历每个分类变量，画出每一个分类变量的条形图
    
plt.tight_layout()  # 自动调整子图布局，避免重叠
plt.show() # 展示整个图形

```

分类变量分布的可视化结果如下：
![示例图片](/分类变量可视化.png)

可以看到，由于有些变量的取值样本过少，可能会导致无法完整地研究该变量的效果。如nasogastric_reflux的slight取值，rectal_exam_faces的serosanguious取值等，几乎没有样本。对比cp_data的不同取值，分布比较均匀，是更理想的变量。对nasogastric_tube和nasogastric_reflux，不同变量取值似乎没有造成结果的很大差异，即不同结果之间的比率呈一个稳定的形态，可以预见它们和目标变量的相关性应该是较小的。


### In[6]

对数值变量的分布进行可视化

```Python
plt.figure(figsize=(14, len(num_cols) * 3))

for i, col in enumerate(num_cols):
    plt.subplot(len(num_cols), 2, i+1)
    sns.histplot(x=col, hue="outcome", data=train, bins=30, kde=True, palette='YlOrRd')
    plt.title(f"{col} distribution for outcome", fontweight="bold")
    plt.ylim(0, train[col].value_counts().max() + 10)
    
plt.tight_layout()
plt.show()

```

数值变量分布的可视化结果如下：
![示例图片](/数值变量可视化.png)

从这些直方图可以看出，有些数据是比较好的接近正态分布的形态，如rectal_temp和packaged_cell_volumn。pulse在存活(lived)情况下有明显右偏，可能意味着较低的取值更容易使马活下来。有几个变量体现出了明显的分类变量特征，如hospital_number, lesion_1, lesion_2, lesion_3，它们的取值集中在一或几个值上，因此可以考虑转化为分类变量。

### In[7]

绘制矩阵散点图，矩阵散点图是一种强大的可视化工具，能够同时展示多个变量之间的两两关系，还可以快速识别变量之间的强相关性和弱相关性，提高数据分析的效率。

```Python
def plot_pair(df_train,num_var,target,plotname):
    '''
    Funtion to make a pairplot:
    df_train: total data
    num_var: a list of numeric variable
    target: target variable
    '''
    g = sns.pairplot(data=df_train, x_vars=num_var, y_vars=num_var, hue=target, corner=True,  palette='YlOrRd')
    g._legend.set_bbox_to_anchor((0.8, 0.7))
    g._legend.set_title(target)
    g._legend.loc = 'upper left'
    g._legend.get_title().set_fontsize(14)
    for item in g._legend.get_texts():
        item.set_fontsize(14)

    plt.suptitle(plotname, ha='center', fontweight='bold', fontsize=25, y=0.98)
    plt.show()

plot_pair(train, num_cols, target, plotname = 'Scatter Matrix with Target')
    # 该图不仅给出了各个数值特征的分布，还能通过散点大致看出这两个数值特征是不是存在线性关系

```

矩阵散点图如下：
![示例图片](矩阵散点图.png)


### In[8]

绘制相关性热力图

```Python
df_encoded = train.copy()
    # 复制原始数据框的副本，避免直接修改原始数据框

categorical_vars = ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 
                    'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 
                    'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 
                    'rectal_exam_feces', 'abdomen', 'abdomo_appearance', 'surgical_lesion', 
                    'cp_data', 'outcome']
    # 定义分类变量


label_encoders = {}
for column in categorical_vars:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(train[column])
    label_encoders[column] = le
    # 将每个分类变量的不同类别转换为整数。

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    # 排除用不到的id列
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    # 排除后的所有列
    corr = df[columns_without_excluded].corr()
    # 计算这些列之间的相关性矩阵
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrBr_r', annot=True, annot_kws={"size": 6})
    plt.title(title_name)
    plt.show()
    # 定义绘制相关性热力图函数


plot_correlation_heatmap(df_encoded, 'Encoded Dataset Correlation')

```

相关性热力图如下：
![示例图片](/相关性热力图.png)

在这张图中，色块越深表明两个变量正相关性越强；色块越浅表明两个变量负相关性越强。正相关性最强的是lesion_2和lesion_3，达到了0.64；其次是surgical_lesion和surgery，达到了0.51。值得注意的是，这两个变量在语义上就有一定关联，比如surgical_lesion表示一种病变是否可用手术治疗，而surgery表示是否在事实上经历了手术。负相关性最强的是total_protein和nasogastric_reflux_ph，达到了-0.58；total_protein和abdomo_protein也达到了-0.47。


## Part 4: 数据清洗和特征工程 (对应单元组为9-11)

### In[9]

这段代码定义了preprocessing和features_engineering两个函数，分别用来进行数据预处理和特征工程。preprocessing函数主要具备以下功能：标签编码、独热编码、值替换、缺失值填充和映射、列删除。而features_engineering函数主要有以下功能：对lesion_2列进行二值化处理，将大于 0 的值设为 1，否则设为 0；创建一个名为 abs_rectal_temp 的新列，计算 rectal_temp 列与 37.8 的差的绝对值；删除 rectal_temp 列。

```python
# 定义preprocessing函数，目的在于对数据进行预处理
def preprocessing(df, le_cols, ohe_cols):
    
    le = LabelEncoder()
    
    for col in le_cols:
        df[col] = le.fit_transform(df[col])
    # 遍历 le_cols 中的每一列，对这些列进行标签编码。将分类变量转化为数值标签
        
    df = pd.get_dummies(df, columns = ohe_cols)
    # 对 ohe_cols 中的每一列进行独热编码。
    
    df["pain"] = df["pain"].replace('slight', 'moderate')
    df["peristalsis"] = df["peristalsis"].replace('distend_small', 'normal')
    df["rectal_exam_feces"] = df["rectal_exam_feces"].replace('serosanguious', 'absent')
    df["nasogastric_reflux"] = df["nasogastric_reflux"].replace('slight', 'none')
    # 对某些列中的特定值进行替换
      
    df["temp_of_extremities"] = df["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
    df["peripheral_pulse"] = df["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
    df["capillary_refill_time"] = df["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
    df["pain"] = df["pain"].fillna("depressed").map({'alert': 0, 'depressed': 1, 'moderate': 2, 'mild_pain': 3, 'severe_pain': 4, 'extreme_pain': 5})
    df["peristalsis"] = df["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
    df["abdominal_distention"] = df["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
    df["nasogastric_tube"] = df["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
    df["nasogastric_reflux"] = df["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
    df["rectal_exam_feces"] = df["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
    df["abdomen"] = df["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
    df["abdomo_appearance"] = df["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})
    # 对某些列进行缺失值填充，并将填充后的值映射为数值
    
    df.drop('lesion_3',axis=1,inplace=True)
    # 删除lesion_3列，因为lession_3列几乎全是0

    return df
    # 返回修改后的数据框

# 定义featuers_engineering函数，进行特征工程
def features_engineering(df):
    df['lesion_2'] = df['lesion_2'].apply(lambda x:1 if x>0 else 0)
    data_preprocessed = df.copy()
     
    data_preprocessed["abs_rectal_temp"] = (data_preprocessed["rectal_temp"] - 37.8).abs()
    data_preprocessed.drop(columns=["rectal_temp"])
    # 创建一个叫abs_rectal_temp的列，用来代替原先的rectal_temp列
    
    return data_preprocessed
    # 返回经过特征工程后的数据框

```


### In[10]

这段代码主要进行了以下操作：首先，定义需要进行标签编码和独热编码的列；然后，调用第九单元组中的proprocessing函数，对训练数据、测试数据和原始训练数据进行预处理；接着，合并训练数据和原始训练数据，并删除重复行，确保数据的一致性和完整性；随后，调用feature_engineering函数，对合并后的数据和测试数据进行特征工程；最后，打印数据框的形状和是否存在缺失值，显示合并后的数据框的前几行。

```Python
le_cols = ["surgery", "age", "surgical_lesion", "cp_data"]
ohe_cols = ["mucous_membrane"]
    # 进行标签编码和独热编码

train = preprocessing(train, le_cols, ohe_cols)
test = preprocessing(test, le_cols, ohe_cols)
train_orig = preprocessing(train_orig, le_cols, ohe_cols)
    # 对训练数据、测试数据和原始训练数据进行预处理
 
total = pd.concat([train, train_orig], ignore_index=True)
total.drop_duplicates(inplace=True)
    # 合并训练数据和原始训练数据

total = features_engineering(total)
test = features_engineering(test)
    # 对合并后的数据和测试数据进行特征工程


print(f'train shape: {train.shape}')
print(f'are there any null values in train: {train.isnull().any().any()}\n')

print(f'test shape: {test.shape}')
print(f'are there any null values in test: {test.isnull().any().any()}\n')

print(f'total shape: {total.shape}')
print(f'are there any null values in total: {total.isnull().any().any()}\n')
    # 打印数据框的形状和是否存在缺失值


```

代码运行结果如下：

train shape: (1235, 32)
are there any null values in train: False

test shape: (824, 32)
are there any null values in test: False

total shape: (1531, 33)
are there any null values in total: True



### In[11]

这段代码的主要功能是对训练数据和测试数据进行缺失值填充，使用 KNNImputer 基于相似性来估计缺失值，确保填充后的数据质量，并通过替换填充后的列来更新数据框，以确保所有列都经过相同的预处理步骤，最终检查填充后是否仍有缺失值，并显示填充后的数据框的前几行，以便查看填充后的结果。相比于均值、中位数、中位数填充等传统填充方式，KNN填充使用了K近邻算法，基于数据点之间的相似性来估计缺失值，可以更好地保留数据的局部结构和特征

```Python
num_cols.remove('lesion_3')
num_cols.append('abs_rectal_temp')
    # 移除和添加数值列


imputer = KNNImputer(n_neighbors=12) # 10 is good 
    # 初始化一个KNNImputer对象，设置 n_neighbors 参数为12，表示使用最近的12个邻居来填充缺失值。

df_train_imputed = pd.DataFrame(imputer.fit_transform(total[num_cols]), columns=num_cols)
df_test_imputed = pd.DataFrame(imputer.transform(test[num_cols]), columns=num_cols)
    # 进行KNN缺失值填充

df_train_null = df_train_imputed[df_train_imputed.isnull().any(axis=1)]
df_test_null = df_test_imputed[df_test_imputed.isnull().any(axis=1)]
    # 检查填充后是否仍有缺失值

print('No. of records with missing value in Train data set after Imputation : {}'.format(df_train_null.shape[0]))
print('No. of records with missing value in Test data set after Imputation : {}'.format(df_test_null.shape[0]))
    # 打印缺失值行数 填充后训练集和测试集记录中是否存在缺失值

print_sl()
    # 调用函数，打印分割线

total_2 = total.drop(num_cols, axis=1).reset_index()
total_2 = pd.concat([total_2, df_train_imputed], axis=1)
    
test_2 = test.drop(num_cols, axis=1).reset_index()
test_2 = pd.concat([test_2, df_test_imputed], axis=1)
    # 替换填充后的列

print('Shape of the Total data set : {}'.format(total_2.shape))
print('Shape of the Test data set : {}'.format(test_2.shape))


```

代码运行结果如下：
No. of records with missing value in Train data set after Imputation : 0
No. of records with missing value in Test data set after Imputation : 0
        ==================================================

Shape of the Total data set : (1531, 34)
Shape of the Test data set : (824, 33)



## Part 5: 建模 (对应单元组为12-18)

### In[12] 

这段代码的主要功能是准备训练数据和测试数据，并进行矩阵化处理。通过删除目标列、转换目标列的类别标签为数值形式、删除中间变量并释放内存，最终生成了特征矩阵 X_train 和 X_test，以及目标矩阵 y_train。

```Python
# 准备训练数据（矩阵化）
X_train = total_2.drop(columns=[target])
    # 删除target列（outcome），生成特征矩阵，为X
y_train = total_2[target].map({'died':0,'euthanized':1,'lived':2})
    # 单独列出target列，为矩阵Y
X_test = test_2
    # 测试数据矩阵化

print(f'X_train shape: {X_train.shape}')

print(f'X_test shape: {X_test.shape}')

print(f'y_train shape: {y_train.shape}')

del train, test, total, test_2, total_2
gc.collect();
    # 删除中间变量并释放内存，减小程序内存


```

代码运行结果如下：

X_train shape: (1531, 33)
X_test shape: (824, 33)
y_train shape: (1531,)


### In[13]

这段代码的主要功能是计算类别权重，以便在模型训练中平衡不同类别的样本。通过获取唯一类别标签、创建类别到索引的映射、将类别标签转换为数值形式、计算每个类别的样本数量、计算总样本数量、计算每个类别的权重，最终生成了一个类别权重字典 class_weights_dict，用于在模型训练中平衡不同类别的样本。

```Python
classes = np.unique(y_train) 
    # 获取唯一类别标签，唯一类别标签为：died lived euthanized
class_to_index = {clas: idx for idx, clas in enumerate(classes)}
    # 创建类别到索引的映射
y_train_numeric = np.array([class_to_index[clas] for clas in y_train])
    # 调用class_to_index，将类别标签转换为数值形式，并存储到一个名为y_train_numeric的numpy数组中

class_counts = np.bincount(y_train_numeric)
    # 计算每个类别的样本数量，并返回一个数组 class_counts，其中每个元素表示对应类别的样本数量。

total_samples = len(y_train_numeric)
    # 计算总样本数量

class_weights = total_samples / (len(classes) * class_counts)
    # 计算每个类别的权重

class_weights_dict = {clas: weight for clas, weight in zip(classes, class_weights)}

```


### In[14]

这段代码定义了一个 Splitter 类，用于将数据集划分为训练集和验证集。通过 KFold 方法进行交叉验证，并支持多个随机种子，以确保划分的随机性和可重复性。最终返回一个生成器，每次迭代返回一组训练集和验证集的划分结果。比起简单划分，使用KFold进行交叉验证，可以显著减少评估偏差，提高模型的泛化能力，同时，相比起留一法交叉验证等其他验证方法，KFold交叉验证有着适中的计算复杂度

```Python
class Splitter:
    # 初始化方法
    def __init__(self, n_splits=5, test_size=0.2):
        # n_splits是数据集的划分次数，而test_size是测试集的比例，通常默认数值为5和0.2
        self.n_splits = n_splits
        self.test_size = test_size
        # 将设定的两个参数赋值给实例变量

    # 定义 split_data 方法
    def split_data(self, X, y, random_state_list):
        for random_state in random_state_list:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val

```


### In[15]

这段代码定义了一个 Classifier 类，用于初始化和定义多个分类模型。通过 _define_model 方法定义多个模型的超参数，并根据设备类型（CPU 或 GPU）调整模型的参数。最终返回一个包含多个分类模型的字典，每个模型使用相应的超参数进行初始化。此外，在这里使用了_define_model方法定义了多个分类模型，对模型进行了封装，隐藏模型的内部实现细节，对外部代码提供一个简单的接口（ self.models），从而提高代码的可读性和可维护性。

```Python
class Classifier:
    # 初始化方法
    def __init__(self, n_estimators=1000, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)

    # 定义 _define_model 方法，用于定义多个分类模型  
    def _define_model(self):
        
        # 定义 XGBoost 模型参数
        # 定义一个字典 xgb_optuna1，包含 XGBoost 模型的超参数。
        xgb_optuna1 = {
            'n_estimators': 500,
            'learning_rate': 0.14825592807938784,
            'booster': 'gbtree',
            'lambda': 8.286104243394034,
            'alpha': 3.218706261523848,
            'subsample': 0.9641392997798903,
            'colsample_bytree': 0.6489144243365093,
            'max_depth': 4, 
            'min_child_weight': 3,
            'eta': 1.230361841253566,
            'gamma': 0.007588382469327802, 
            'grow_policy': 'depthwise',
            'objective': 'multi:softmax',
          #  'class_weight': class_weights_dict,
            'random_state': self.random_state,
        }
        
        # 定义一个字典 xgb1_params，包含 另一个XGBoost 模型的超参数。
        xgb1_params = {
            "n_estimators": 1000,
            "max_depth": 3,
            "learning_rate": 0.55, 
            "min_child_weight": 2,
            "colsample_bytree": 0.9, 
            "objective": "multi:softmax", 
            "eval_metric": "merror",
         #   'class_weight': class_weights_dict,
            "random_state": self.random_state, 
        }
        
        # 定义一个字典 xgb_params，包含另一个 XGBoost 模型的超参数。
        xgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'n_jobs': -1,
            'eval_metric': 'merror',
            'objective': 'multi:softmax',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
            'class_weight':class_weights_dict,
        }

        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        # 定义一个字典 lgb_optuna1，包含 LightGBM 模型的超参数。
        lgb_optuna1 = {
            'num_iterations': 200,
            'learning_rate': 0.05087818591635374,
            'max_depth': 10,
            'alpha': 4.34921696876783,
            'subsample': 0.512929283477029,
            'colsample_bytree': 0.5421760951211009, 
            'min_child_weight': 4,
            'random_state': self.random_state,
            'objective': 'multiclass',
            'class_weight':class_weights_dict,
            'verbose': -1,
        }
        
        # 定义一个字典 lgb_params2，包含另一个 LightGBM 模型的超参数。
        lgb_params2 = {
            'n_estimators': self.n_estimators,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'class_weight':class_weights_dict,
            'verbose': -1,

        }

        # 定义一个字典 cat_optuna1，包含 CatBoost 模型的超参数。
        cat_optuna1 = {
            'iterations': 700,          
            'learning_rate': 0.06806932341035855,
            'depth': 3,
            'l2_leaf_reg': 4.246994639881441,
            'bagging_temperature': 0.08262764367292164,
            'random_strength': 6.922710769000274, 
            'border_count': 88,
            'random_state': self.random_state,
            'verbose': False,
        }

        # 定义一个字典 hist_params，包含 HistGradientBoosting 模型的超参数。
        hist_params = {
            'l2_regularization': 0.01,
            'early_stopping': True,
            'learning_rate': 0.01,
            'max_iter': self.n_estimators,
            'max_depth': 4,
            'max_bins': 255,
            'min_samples_leaf': 10,
            'max_leaf_nodes':10,
            'class_weight':'balanced',
            'random_state': self.random_state
        }
        best_hyperparams_xgb = {
            'eta': 0.2734096744203229, 
            'n_estimators': 251, 
            'max_depth': 1,
              'reg_lambda': 1.3536521735953297,
                'subsample': 0.9372043032806799,
                  'min_child_weight': 5, 
                  'colsample_bytree': 0.32973413695986586,
                    'objective': 'multi:softmax'
        }
        best_hyperparams_lgbm = {
            'n_estimators': 146,
              'learning_rate': 0.09732455260435911, 
              'max_depth': 8, 
              'num_leaves': 973, 
              'reg_lambda': 5.558974411222393,
                'reg_alpha': 5.94913795893992,
                'subsample': 0.057493821911338956,
                'colsample_bytree': 0.7716515051686431,
                'min_child_samples': 46,
                'min_child_weight': 7,
                'objective': 'multiclass',
                'metric':'multi_logloss',
                'boosting_type': 'gbdt',
                'verbose':-1,
        }
        best_hyperparams_cb = {
            'iterations': 210, 
            'learning_rate': 0.21569043805753133,
              'depth': 3,
                'l2_leaf_reg': 6.171143053175511, 
                'grow_policy': 'Depthwise',
                  'bootstrap_type': 'Bayesian',
                    'od_type': 'Iter', 
                    'eval_metric': 'TotalF1',
                      'loss_function': 'MultiClass',
                        'random_state': 42,
                          'verbose': 0
                          
        }

        # 定义模型字典，models，包含多个分类模型，每个模型使用相应的超参数进行初始化。
        models = {
            'xgb1': xgb.XGBClassifier(**xgb_optuna1),
            #'lgb1': lgb.LGBMClassifier(**lgb_optuna1),
            'hgb': HistGradientBoostingClassifier(**hist_params),
            #'cat1': CatBoostClassifier(**cat_optuna1),
            'xgb2': xgb.XGBClassifier(random_state=self.random_state),
            #'xgb3': xgb.XGBClassifier(**xgb1_params),
            'xgb4': xgb.XGBClassifier(**xgb_params),
            #'lgb2': lgb.LGBMClassifier(**lgb_params2),
            'cat2': CatBoostClassifier(random_state=self.random_state,verbose=False),
            #'rf': RandomForestClassifier(random_state=self.random_state),
            'xgb5':xgb.XGBClassifier(**best_hyperparams_xgb),
            'lgb3': lgb.LGBMClassifier(**best_hyperparams_lgbm),
            #'cat3':CatBoostClassifier(**best_hyperparams_cb)
        }
        
        return models

```


### In[16]

这段代码定义了一个 OptunaWeights 类，用于使用 Optuna 库进行模型权重优化，并通过加权平均法组合多个模型的预测结果。通过集成多个模型并优化权重，可以提高整体的预测性能，减少单个模型的偏差和方差，提高模型的鲁棒性和泛化能力。此外，同单元组15，封装功能提高了代码的可维护性和可读性，模块化的代码结构也便于扩展和修改。

```Python
class OptunaWeights:
    def __init__(self, random_state, n_trials=1000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-12, 2) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=weights)
        
        weighted_pred_labels = np.argmax(weighted_pred, axis=1)
        f1_micro_score = f1_score(y_true, weighted_pred_labels, average='micro')
        return f1_micro_score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights

```


### In[17]

这段代码的主要功能是通过交叉验证和集成学习来训练多个分类模型，并使用 Optuna 优化模型权重，以提高集成模型的性能。通过初始化参数和实例、进行交叉验证、训练和评估模型、优化模型权重以及预测测试集，最终实现了使用优化后的权重进行预测的功能。这几部分也是所有代码中最核心的部分，集成学习最重要的体现，通过 Classifier 类定义了多个分类模型（如 XGBoost、LightGBM、CatBoost 等），并将这些模型存储在 models 字典中，在交叉验证的每个折叠中，在每个折叠中，使用 Optuna 优化模型权重，并通过加权平均法组合多个模型的预测结果。
具体来说，OptunaWeights 类中的 _objective 方法定义了目标函数，用于计算加权平均预测结果的 F1 微平均分数。fit 方法使用 Optuna 优化模型权重，predict 方法使用优化后的权重进行预测。在每个折叠中，使用优化后的权重对测试数据进行预测，并将预测结果累加到 test_predss 中，最终得到测试数据的集成预测结果。集成学习最大的两个特征“多模型组合”和“模型组合策略”在这里体现的淋漓尽致。

```Python
n_splits = 5
random_state = 42
random_state_list = [42] 
n_estimators = 999 
early_stopping_rounds = 333
verbose = False
device = 'cpu'
splitter = Splitter(n_splits=n_splits)

# 初始化一个用于存储测试预测的数组
test_predss = np.zeros((X_test.shape[0], 3))
ensemble_f1_score = []
weights = []
trained_models = {'xgb':[], 'lgb':[], 'cat':[]}
    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # 获取一组回归模型
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    
    # 初始化列表以存储每个基模型的袋外预测和测试预测
    oof_preds = []
    test_preds = []
    
    # 遍历每个基模型并将其拟合到训练数据，在验证数据上进行评估，并存储预测结果
    for name, model in models.items():
        if ('xgb' in name) or ('lgb' in name) or ('cat' in name)  :
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
        
        test_pred = model.predict_proba(X_test)
        y_val_pred = model.predict_proba(X_val)

        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        f1_micro_score = f1_score(y_val, y_val_pred_labels, average='micro')
        
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] F1 Micro Score: {f1_micro_score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
    
    # 使用 Optuna 找到最佳集成权重
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val, oof_preds)
    
    score = log_loss(y_val, y_val_pred)
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    f1_micro_score = f1_score(y_val, y_val_pred_labels, average='micro')
    
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ---------------> F1 Micro Score: {f1_micro_score:.5f}')
    print_sl()
    
    ensemble_f1_score.append(f1_micro_score)
    weights.append(optweights.weights)
    
    # 使用最佳集成权重对 X_test 进行预测
    _test_preds = optweights.predict(test_preds)
    test_predss += _test_preds / (n_splits * len(random_state_list))
    
    gc.collect()

```

代码运行结果如下：

xgb1 [FOLD-0 SEED-42] F1 Micro Score: 0.74267
hgb [FOLD-0 SEED-42] F1 Micro Score: 0.72638
xgb2 [FOLD-0 SEED-42] F1 Micro Score: 0.72964
xgb3 [FOLD-0 SEED-42] F1 Micro Score: 0.73290
xgb4 [FOLD-0 SEED-42] F1 Micro Score: 0.79153
cat2 [FOLD-0 SEED-42] F1 Micro Score: 0.77850
xgb5 [FOLD-0 SEED-42] F1 Micro Score: 0.74919
lgb3 [FOLD-0 SEED-42] F1 Micro Score: 0.74919
Ensemble [FOLD-0 SEED-42] ---------------> F1 Micro Score: 0.79479
    ----------------------------------------
xgb1 [FOLD-1 SEED-42] F1 Micro Score: 0.71895
hgb [FOLD-1 SEED-42] F1 Micro Score: 0.69281
xgb2 [FOLD-1 SEED-42] F1 Micro Score: 0.72876
xgb3 [FOLD-1 SEED-42] F1 Micro Score: 0.71242
xgb4 [FOLD-1 SEED-42] F1 Micro Score: 0.70261
cat2 [FOLD-1 SEED-42] F1 Micro Score: 0.69935
xgb5 [FOLD-1 SEED-42] F1 Micro Score: 0.70588
lgb3 [FOLD-1 SEED-42] F1 Micro Score: 0.71242
Ensemble [FOLD-1 SEED-42] ---------------> F1 Micro Score: 0.74183
    ----------------------------------------
xgb1 [FOLD-2 SEED-42] F1 Micro Score: 0.70261
hgb [FOLD-2 SEED-42] F1 Micro Score: 0.71895
xgb2 [FOLD-2 SEED-42] F1 Micro Score: 0.74837
xgb3 [FOLD-2 SEED-42] F1 Micro Score: 0.70915
xgb4 [FOLD-2 SEED-42] F1 Micro Score: 0.75163
cat2 [FOLD-2 SEED-42] F1 Micro Score: 0.72222
xgb5 [FOLD-2 SEED-42] F1 Micro Score: 0.73529
lgb3 [FOLD-2 SEED-42] F1 Micro Score: 0.74183
Ensemble [FOLD-2 SEED-42] ---------------> F1 Micro Score: 0.75490
    ----------------------------------------
xgb1 [FOLD-3 SEED-42] F1 Micro Score: 0.78431
hgb [FOLD-3 SEED-42] F1 Micro Score: 0.73203
xgb2 [FOLD-3 SEED-42] F1 Micro Score: 0.76797
xgb3 [FOLD-3 SEED-42] F1 Micro Score: 0.75817
xgb4 [FOLD-3 SEED-42] F1 Micro Score: 0.78758
cat2 [FOLD-3 SEED-42] F1 Micro Score: 0.75163
xgb5 [FOLD-3 SEED-42] F1 Micro Score: 0.75817
lgb3 [FOLD-3 SEED-42] F1 Micro Score: 0.77124
Ensemble [FOLD-3 SEED-42] ---------------> F1 Micro Score: 0.79739
    ----------------------------------------
xgb1 [FOLD-4 SEED-42] F1 Micro Score: 0.70588
hgb [FOLD-4 SEED-42] F1 Micro Score: 0.70261
xgb2 [FOLD-4 SEED-42] F1 Micro Score: 0.70261
xgb3 [FOLD-4 SEED-42] F1 Micro Score: 0.72222
xgb4 [FOLD-4 SEED-42] F1 Micro Score: 0.72876
cat2 [FOLD-4 SEED-42] F1 Micro Score: 0.67974
xgb5 [FOLD-4 SEED-42] F1 Micro Score: 0.69281
lgb3 [FOLD-4 SEED-42] F1 Micro Score: 0.71242
Ensemble [FOLD-4 SEED-42] ---------------> F1 Micro Score: 0.75490



### In[18]

这段代码的主要功能是计算集成模型的平均 F1 微平均分数，并打印每个模型的平均权重及其标准差。通过计算平均值和标准差，来评估集成模型的性能和每个模型在集成中的重要性。

```Python
# 计算集成模型的平均对数损失分数
mean_score = np.mean(ensemble_f1_score)
std_score = np.std(ensemble_f1_score)
print(f'Ensemble F1 score {mean_score:.5f} ± {std_score:.5f}')

# 打印每个模型的集成权重平均值和标准差
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')

```

代码运行结果如下：

Ensemble F1 score 0.76485 +0.02712
--- Model Weights --
xgb1: 0.57301 ± 0.41870
hgb: 0.97677 ± 0.23297
xgb2: 0.93762 ± 0.62647
xgb3: 0.86145 ± 0.62577
xgb4: 1.08197 ± 0.60358
cat2: 0.79954 ± 0.36700
xgb5: 1.03261 ± 0.54970
lgb3: 0.59779 ± 0.43867


## Part 6：预测和提交（对应单元组为19-21）

### In[19]

这段代码的主要功能是对测试集的预测结果进行后处理，通过调整预测概率来修正模型的预测。具体来说，代码通过三个条件判断来调整预测结果，确保预测结果符合特定的逻辑和阈值要求。通过这些后处理步骤，可以提高模型的预测准确性和可靠性。

下面是对这段较为“抽象”的代码的详细解释：
单个预测（由pred表示）看起来像这样：[0.35, 0.25, 0.40]，其中，pred[0]是马dead的概率，pred[1]是马euthanized的概率，pred[2]是马lived的概率。

第一个条件检查马存活的概率是否大于被安乐死的概率，同时检查马死亡的概率是否小于被安乐死和存活的概率之和。在“马存活的概率大于马被安乐死的概率，且马死亡的概率小于另外两种情况之和”的情况下，我们预测马的结局是存活。本质上，我们处理像[0.4, 0.21, 0.39]或[0.45, 0.15, 0.4]这样的模糊情况，模型会预测马死亡，但是通过这个约束后会被调整为存活。

在下一个条件中，检查结局是否更有可能是死亡而不是存活或安乐死，然而，如果死亡和安乐死的概率之间的差异低于阈值，我们预测结局为安乐死。顺便一提，为什么差异阈值是0.3？在仅包含合成数据的分布中，两者概率之差为0.13，所以，阈值起码要大于0.13，否则，过多的预测结果将被调整为安乐死；同时，活着的马应该是最多的且不大于或者略大于0.5，因此，死亡和安乐死的概率之差一定不会大于0.5，在0.13到0.5这个范围内，我们简单地（或者说直觉地）选择两者的均值（0.315），然后近似地认为是0.3.

最后一个条件调整了预测马结局为存活的阈值。

```Python
for pred in test_predss:
    if (pred[1] < pred[2]) and ((pred[2] + pred[1]) > pred[0]): 
        pred[0] = 0
        pred[1] = 0
        pred[2] = 1
    # 这个条件的目的是确保在类别1和类别2的概率之和大于类别0的概率时，将预测结果修正为类别2。
        
    if (pred[0] > pred[2]) and (pred[0] > pred[1]) and (pred[0] - pred[1] < 0.3): 
        pred[0] = 0
        pred[1] = 1
        pred[2] = 0
    # 这个条件的目的是确保在类别0的概率较大且与类别1的概率差异较小时，将预测结果修正为类别1。
        
    if pred[2] > 0.42:
        pred[0] = 0
        pred[1] = 0
        pred[2] = 1
    # 这个条件的目的是确保在类别2的概率较大时，将预测结果修正为类别2。

```


### In[20]

这段代码的功能是生成提交文件，并给出生成的数据框

```Python
submission = pd.DataFrame({'id': sample_submission['id'], 'outcome': np.argmax(test_predss, axis=1)})
submission['outcome'] = submission['outcome'].map({0:'died',1:'euthanized',2:'lived'})
submission.to_csv('submission.csv',index=False)
submission
```


### In[21]

调用plot_count函数，给出最终预测的可视化结果
```Python
plot_count(submission, 'outcome', 'Predicted Variable(Outcome) Distribution')
```

最终预测的可视化结果如下：

![示例图片](/结果预测.png)


最终得分如下：
![示例图片](/最终得分.png)


## Part 7: 感想

### 关于代码和报告

关于代码，这是我们有史以来写过的最长的也是最复杂的代码，为了完成这个极具挑战性的项目，我们查阅了大量的资料和notebook。在kaggle上读了很多有意义的notebook和comments，并对每一份notebook都进行了剖析，对比了他们之间的优缺点，不管是在数据处理方法上还是模型建立上，然后对不同方案进行了测试，最终得到了我们认为的最优的方案。当然，写代码的过程更是难中之难，很多机器学习相关的高级库（例如Optuna），这是我们从来没有接触过的，这份代码中很多很多知识都是我们“现学现卖”的。

关于报告，一开始我们并不知道如何写这份报告，我们甚至没有余力去考虑怎么写报告，因为单单代码部分对于我们来说就是挑战性十足的任务。但是当我们看到kaggle上的notebook之后，我们顿时觉得，如果我们也能写出一份图文并茂的notebook，那是一件相当有成就感的事情，而且，清晰生动的报告也会为我们的项目增加更多的可读性。于是，为了将文字和代码整合在一起，我们自学了markdown；尽管我们有使用matplotlib的经验，但是为了画出更生动直观的图，我们自学了seaborn。最终得到了这份在我们看来图文（还有代码（doge））并茂的报告。

### 关于这门课程
实际上，我们团队的两位成员都是来自人文社科而非理工科，根据新课推荐中的提示，更推荐人文社科的同学选修数据科学导引（C），但是由于我们对机器学习和数据科学有着浓厚的兴趣，我们选择了这一门难度较大的数据科学导引（B）。通过这门课程，我们不仅学到了课上的理论知识，而且对于机器学习实战有了一定的经验。除此之外，就像在前面说的，除了课程内容本身，我们也花了很多课下时间去学习课外的知识。我们发现，相比单纯地了解这些数据分析过程，自己动手做项目带来的收获是更有实际价值和效率的。在短短两周内，我们的综合数据处理能力得到了很大提升。总的来说，这门课程不仅让我们从理论上入门机器学习，更是让我们真正“动手”，从实践和代码上入门了机器学习。