
# [Titanic:\_Machine_Learning_from_Disaster](https://www.kaggle.com/c/titanic)

## 结果

训练集最高得分：0.801372，模型：决策树(DecisionTree)，测试集得分为：0.77033<br>
![myscole](https://github.com/incipient1/titanic_machinelearn_from_disaster/blob/master/test_score_kaggle.PNG)

## 数据清洗


```python
import pandas as pd
import numpy as np
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

%matplotlib inline
```


```python
titanic = pd.read_csv('train.csv')
s = '平均生存概率：{0:.2f} %'.format(titanic.describe().loc['mean','Survived'] * 100)
s
```




    '平均生存概率：38.38 %'



假设在`'Survived'`中全部填零，预测的正确率差不多也有`61.61% == 1 - 平均生存概率`<br>
将列`Sex`标签化


```python
titanic['sex'] = titanic['Sex']
titanic.loc[titanic['sex'] == 'female','sex'] = 0
titanic.loc[titanic['sex'] == 'male','sex'] = 1
titanic.sample(6)  # 选6行数据看看
```




`.loc(行索引,列索引)`返回满足索引条件的值(只有一个时)或者DataFrame，默认优先采用自定义索引，无自定义索引时采用系统索引，相反的是`iloc`

将Sex中的值数字化，也可以采用LabelEncoder，代码如下：
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(titanic['sex'])
le.transform(titanic['sex'])
```
但是用le有可能指定male为0，所以手动指定<br>
查看数据整体情况


```python
titanic.describe() # 仅展示值为数字的列
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



`.describe()`输出的结果仍然是`pandas.core.frame.DataFrame`对象


```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    sex            891 non-null object
    dtypes: float64(2), int64(5), object(6)
    memory usage: 90.6+ KB


`Age`这列中有NaN，需要单独处理NaN的值；<br>
`Embarked`这一列有2个空值，需要单独处理；<br>
`.info()`生成的仍然是`Nonetype`对象，所以不能对它进行切片、定位操作；
### 生成`Title`（称谓）列


```python
titanic['Title'] = titanic.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
pd.crosstab(titanic['Title'],titanic['Sex'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



`.extract(' ([A-Za-z]+)\.',expand=False)`从字符串中提取出符合正则表达式`' ([A-Za-z]+)\.'`的字符串，目的是从`Name`中把称谓提取出来；<br>
`pandas.crosstab(index,columns)`默认计算两个或多个factor(因子)出现的频次。Title中值不一样但是意思一样的，需要合并，例如Mme和Mrs，Mlle、Ms和Miss；出现频次很少，意思又不一样的，统一成一类。


```python
titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic['Title'] = titanic['Title'].replace(['Mlle','Ms'], 'Miss')
titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')
```

`.repalce(消失值,返回值)`<br>
将Title标签化


```python
titanic['title'] = titanic['Title']
titanic.loc[titanic['Title'] == 'Master','title'] = 0
titanic.loc[titanic['Title'] == 'Miss','title'] = 1
titanic.loc[titanic['Title'] == 'Mr','title'] = 2
titanic.loc[titanic['Title'] == 'Mrs','title'] = 3
titanic.loc[titanic['Title'] == 'Rare','title'] = 4
```

将Embarked（登船港口）标签化

```python
titanic['embarked'] = titanic['Embarked']
titanic['embarked'] = titanic['embarked'].replace('S',int(0))
titanic['embarked'] = titanic['embarked'].replace('C',int(1))
titanic['embarked'] = titanic['embarked'].replace('Q',int(2))
titanic['embarked'].fillna(3,inplace=True)
```

`.fillna()`用新值替换值为空NaN的，当然上面也可以采用`.loc`的方法<br>
探索Embarked和生存是否有关系


```python
titanic[['embarked','Survived']].groupby(
    'embarked',as_index=False).mean().sort_values(by='Survived',ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.336957</td>
    </tr>
  </tbody>
</table>
</div>



生存率在不同登船口岸中也有差异；<br>
`.groupby()`分类汇总，返回一个DF对象；`.mean()`DF中的值为每个类所有值的期望；`sort_values(by,axis=0,ascending=True)`根据by这一列在axis轴进行升序排序。


```python
titanic['embarked'].count()
```




    891



经过替换，embarked中无空值NaN，因为count()不统计空值
### 处理age中的空值


```python
guess_ages = np.zeros((2,3))
for i in range(2):
    for j in range(3):
        guess_df = titanic[(titanic['sex']==i) &
                           (titanic['Pclass']==j+1)]['Age'].dropna()
        age_guess = guess_df.median()
        guess_ages[i,j] = int( age_guess/0.5 +0.5 ) * 0.5

guess_ages
```




    array([[ 35. ,  28. ,  21.5],
           [ 40. ,  30. ,  25. ]])



根据sex、Pclass分类，得出每个类中的年龄的中位数，然后再用中位数去代替Age中的空值；


```python
for i in range(2):
    for j in range(3):
        titanic.loc[(titanic.Age.isnull() & (titanic['sex']==i) &
                           (titanic['Pclass']==j+1),'Age')] = guess_ages[i,j]

```


```python
titanic.age = titanic.Age.astype(int)
```


```python
titanic.loc[titanic.age <=12 ,'age'] = 1
titanic.loc[(titanic.age >12) & (titanic.age <= 18),'age'] = 2
titanic.loc[(titanic.age >18) & (titanic.age <= 32),'age'] = 3
titanic.loc[(titanic.age >32) & (titanic.age <= 60),'age'] = 4
titanic.loc[(titanic.age >60) & (titanic.age <= 80),'age'] = 5
```


```python
titanic[['age','Survived']].groupby('age',as_index=False).mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.579710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.419708</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.227273</td>
    </tr>
  </tbody>
</table>
</div>



不同年龄段生存率是有差异的，壮年段(3)的生存率低于平均值，其它都比平均值高，体现了人性的光辉

### 增加标签isalone

```python
titanic['isalone'] = 0
titanic.loc[(titanic['Parch'] + titanic['SibSp']) == 0,'isalone'] = 1
```

`isalone`在船上是否孤身一人，如果它的父母孩子(Parch)数量与亲戚(SibSp)数量和为0，那么其为孤身


```python
titanic[['isalone','Survived']].groupby('isalone',as_index=False).mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isalone</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.505650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.303538</td>
    </tr>
  </tbody>
</table>
</div>



孤身一人的生存概率低
### 增加标签age\*pclass

```python
titanic['age*pclass'] = titanic['age'] * titanic.Pclass
```


```python
titanic.Pclass.value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64




```python
titanic['age'].value_counts()
```




    3.0    456
    4.0    274
    2.0     70
    1.0     69
    5.0     22
    Name: age, dtype: int64




```python
titanic[['age*pclass','Survived']].groupby(
    'age*pclass',as_index=False).mean().sort_values(by='Survived',ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age*pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.965517</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.602740</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.570000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>0.412698</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>0.370370</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.0</td>
      <td>0.247619</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.214286</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.0</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12.0</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>



`age*pclass`将年龄和地位单独特征放大了，年龄大，地位低的存活率低


```python
fareband_count = 10
titanic['fareband'] = pd.qcut(titanic['Fare'],fareband_count)

fare_line_data = titanic[['fareband','Survived']].groupby('fareband',\
as_index=False).mean()

fare_line_data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fareband</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.001, 7.55]</td>
      <td>0.141304</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(7.55, 7.854]</td>
      <td>0.298851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(7.854, 8.05]</td>
      <td>0.179245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(8.05, 10.5]</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(10.5, 14.454]</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(14.454, 21.679]</td>
      <td>0.420455</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(21.679, 27.0]</td>
      <td>0.516854</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(27.0, 39.688]</td>
      <td>0.373626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(39.688, 77.958]</td>
      <td>0.528090</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(77.958, 512.329]</td>
      <td>0.758621</td>
    </tr>
  </tbody>
</table>
</div>



`pandas.qcut()`将维度为1的数据根据值排序后，再切成份，每份同样本数量。返回结果中的值`(-0.001, 7.55]`是`pandas.interval`对象，它有以下属性：
* closed  哪一侧的数被包含在内
* closed_left、open_left	、open_right、closed_right  返回bool值
* left、right	返回左侧值，是一个numpy数，需要转成python中的数才能互相使用
* mid	返回每个区间的中位数


```python
titanic.fareband_num = titanic['Fare'].astype('float')
```


```python
for i in range(fareband_count-1):
    left = fare_line_data.loc[i,'fareband'].left
    left = float(left)
    right = fare_line_data.loc[i,'fareband'].right
    right = float(right)
    titanic.loc[(titanic.fareband_num > left) &
                (titanic.fareband_num <= right),'fareband_num'] = 1 + i

titanic.loc[titanic.fareband_num > right,'fareband_num'] = fareband_count
```

分段完成后将每段参数化；<br>
因为分段时右边界为closed(']'),为了防止在分界点处小数点的影响，所以在分界的最右应单独处理；<br>
整体趋势：船费越贵，存活率越高


```python
titanic[['fareband_num','Survived']].groupby('fareband_num',as_index=False).mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fareband_num</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.141304</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.310811</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>0.184874</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.454545</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>0.404255</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>0.373626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>0.511628</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>0.766667</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = titanic[['Survived','SibSp','Parch','sex','title',
             'embarked','isalone','age*pclass','fareband_num']]
y = y.corr()
y
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>embarked</th>
      <th>isalone</th>
      <th>age*pclass</th>
      <th>fareband_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>1.000000</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.118026</td>
      <td>-0.203367</td>
      <td>-0.365466</td>
      <td>0.323214</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.035322</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>-0.063794</td>
      <td>-0.584471</td>
      <td>-0.174199</td>
      <td>0.358643</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.081629</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>-0.082144</td>
      <td>-0.583398</td>
      <td>-0.146568</td>
      <td>0.369870</td>
    </tr>
    <tr>
      <th>embarked</th>
      <td>0.118026</td>
      <td>-0.063794</td>
      <td>-0.082144</td>
      <td>1.000000</td>
      <td>0.025927</td>
      <td>0.000494</td>
      <td>-0.037441</td>
    </tr>
    <tr>
      <th>isalone</th>
      <td>-0.203367</td>
      <td>-0.584471</td>
      <td>-0.583398</td>
      <td>0.025927</td>
      <td>1.000000</td>
      <td>0.310023</td>
      <td>-0.538914</td>
    </tr>
    <tr>
      <th>age*pclass</th>
      <td>-0.365466</td>
      <td>-0.174199</td>
      <td>-0.146568</td>
      <td>0.000494</td>
      <td>0.310023</td>
      <td>1.000000</td>
      <td>-0.644792</td>
    </tr>
    <tr>
      <th>fareband_num</th>
      <td>0.323214</td>
      <td>0.358643</td>
      <td>0.369870</td>
      <td>-0.037441</td>
      <td>-0.538914</td>
      <td>-0.644792</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



`.corr()`Pearson相关系数，形成相关系数矩阵，默认的是皮尔逊相关系数`method='pearson'`，返回值的范围是：[-1,1]。还有其它的方法：`'kendall'、'spearman'`。皮尔逊相关系数值对应的相关性程度如下：

|返回值的绝对值|相关性程度|
|------|-------|
|0.8 ~ 1.0|极强|
|0.6 ~ 0.8|强|
|0.4 ~ 0.6|中|
|0.2 ~ 0.4|弱|
|0   ~ 0.2|极弱，无关|


```python
sns.set(style='whitegrid',font_scale=0.6,font='SimHei')
matplotlib.rcParams['axes.unicode_minus']=False
plt.subplots_adjust(left=0.15,right=0.9,
                    bottom=0.1, top=0.9)

grid_kws = {"height_ratios": (.85, .05), "hspace": .3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
mask = np.zeros_like(y)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(y,mask=mask,square=False,vmin=-1,vmax=1,\
            robust=True,annot=True,center=1,linewidth=0.5,fmt='>5.2f',ax=ax,
            cbar_ax=cbar_ax,cbar_kws={"orientation": "horizontal"})

plt.savefig('titanic_pearson.jpg',dpi=600)
```


    <matplotlib.figure.Figure at 0x269528100b8>



![各特征之间的关系](https://github.com/incipient1/titanic_machinelearn_from_disaster/blob/master/img/output_47_1.png)


* 整体设置图的基本情况
```
sns.set(
context='notebook',
style='darkgrid',        # 背景模式，可选：darkgrid黑色网格, whitegrid白色网格,
                         # dark,white, ticks
palette='deep',          # 调色板
font='sans-serif',       # 字体
font_scale=1,            # 字体缩放大小
color_codes=False,
rc=None               # 重设其它系统默认设置，例如更改系统默认的字体：
                      # plt.rcParams['font.sans-serif']=['SimHei']
)
```
`matplotlib.rcParams['axes.unicode_minus']=False`正常显示标签中的负号
```
plt.subplots_adjust(left=0.15,  # 子图(x、y轴围成的那个矩形)左边开始位置，整张图的宽度为1
                                # 因为轴标签fareband_num比较长，所以要单独设置
                    right=0.9,  # 子图右边到哪里停止
                    bottom=0.1, # 下边从哪里开始
                    top=0.9 )   # 高度结束位置
```

* 将colorbar 水平放置，直接从官网中抄下来的
```
grid_kws = {"height_ratios": (.85, .05), "hspace": .3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
mask = np.zeros_like(y)
mask[np.triu_indices_from(mask)] = True
```
* 绘制热力图
```
sns.heatmap(
y,             # 上面生成的皮尔逊相关系数组成的DF
mask=mask,     # 遮挡生成的图
square=False,  # 是否保证每个小图为正方形
vmin=-1, vmax=1,  # colorbar 的最小值、最大值
robust=True,
annot=True,       # 是否把数据标到每个小图中
center=1,         # colorbar中颜色最深的位置其实我想实现的是-1为蓝，0为无色，1为绿色，未果
linewidth=0.5,    # 图中网格线的线宽
fmt='>5.2f',      # 标注的数字的格式：右对齐5个字符保留两位小数
ax=ax,
cbar_ax=cbar_ax,
cbar_kws={"orientation": "horizontal"}) # 方向：水平
```
* 保存图片  plt.savefig('图片保存位置和名字/格式',dpi=600)

## 建模


```python
x = titanic[['Pclass','SibSp','Parch','sex','title',
             'embarked','isalone','age*pclass','fareband_num']]
```


```python
from sklearn import tree,linear_model,neighbors,ensemble


dt = tree.DecisionTreeClassifier()
lm = linear_model.LogisticRegression()
knn = neighbors.KNeighborsClassifier(5,weights='uniform')
rf = ensemble.RandomForestClassifier(10)

from sklearn.model_selection import cross_val_score

score_dt = cross_val_score(dt,x,titanic['Survived'],cv=5,scoring='accuracy')
score_dt = np.mean(score_dt)

score_lm = cross_val_score(lm,x,titanic['Survived'],cv=5,scoring='accuracy')
score_lm = np.mean(score_lm)

score_knn = cross_val_score(knn,x,titanic['Survived'],cv=5,scoring='accuracy')
score_knn = np.mean(score_knn)

score_rf = cross_val_score(knn,x,titanic['Survived'],cv=5,scoring='accuracy')
score_rf = np.mean(score_rf)
```


```python
models = pd.DataFrame({'Model':['Decision Tree','LogisticRegression','KNeighbors',
                                'RandomForest'],
                      'Score':[score_dt,score_lm,score_knn,score_rf]})
models.sort_values(by='Score')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>LogisticRegression</td>
      <td>0.785654</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNeighbors</td>
      <td>0.799163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForest</td>
      <td>0.799163</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Decision Tree</td>
      <td>0.804718</td>
    </tr>
  </tbody>
</table>
</div>



**DecisionTree**的模型得分最高


```python
model = dt.fit(x,titanic['Survived'])
```

预测test的数据就用这个模型model

## 预测test
数据导入、清洗同上


```python
test = pd.read_csv('test.csv')
```


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB


注意Fare中有**1个**NaN，train中的Fare没有NaN，一定要处理


```python
test['sex'] = test['Sex']
test.loc[test['sex'] == 'female','sex'] = 0
test.loc[test['sex'] == 'male','sex'] = 1
```


```python
test_guess_age = np.zeros((2,3))
```


```python
for i in range(2):
    for j in range(3):
        test_df = test[(test['sex'] == i) &
                           (test['Pclass'] == j+1)]['Age'].dropna()
        test_df_age = test_df.median()
        test_guess_age[i,j] = int(test_df_age / 0.5 + 0.5) + 0.5

test_guess_age
```




    array([[ 82.5,  48.5,  44.5],
           [ 84.5,  56.5,  48.5]])




```python
test['age'] = test['Age']
```


```python
for i in range(2):
    for j in range(3):
        test.loc[test.age.isnull() & (test['sex'] == i) &
                           (test['Pclass'] == j+1),'age' ] = test_guess_age[i,j]


```


```python
test.age.count()
```




    418




```python
test.loc[test.age <=12,'age'] = 1
test.loc[(test.age > 12) & (test.age <= 18),'age'] = 2
test.loc[(test.age > 18) & (test.age <= 32),'age'] = 3
test.loc[(test.age > 32) & (test.age <= 60),'age'] = 4
test.loc[(test.age > 60) ,'age'] = 5
```


```python
test.loc[(test.age > 60) ,'age'] = 5
```


```python
test['age'].value_counts()
```




    4.0    186
    3.0    158
    2.0     29
    1.0     25
    5.0     20
    Name: age, dtype: int64




```python
test['Title'] = test.Name.str.extract(' ([a-zA-Z]+?)\.',expand=False)
```


```python
pd.crosstab(test['Title'],test['Sex'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Dona</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>240</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>72</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
test['Title'] = test['Title'].replace(['Mlle','Ms'], 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')
```


```python
test['title'] = test['Title']
```


```python
test.loc[test['title'] == 'Mr','title'] = 2
test.loc[test['title'] == 'Miss','title'] = 1
test.loc[test['title'] == 'Mrs','title'] = 3
test.loc[test['title'] == 'Master','title'] = 0
test.loc[test['title'] == 'Rare','title'] = 4
```


```python
test['embarked'] = test['Embarked']
```


```python
test.loc[test['embarked'] == 'S','embarked'] = 0
test.loc[test['embarked'] == 'C','embarked'] = 1
test.loc[test['embarked'] == 'Q','embarked'] = 2
```


```python
test['isalone'] = 0
```


```python
test.loc[(test['Parch'] == 0) & (test['SibSp'] == 0),'isalone'] = 1
```


```python
test['age*pclass'] = test.age * test['Pclass']
```


```python
test['fareband_num'] = test['Fare'].astype('float32')
```


```python
titanic['fareband_'] = pd.qcut(titanic['Fare'],fareband_count)

fare_line_data = titanic[['fareband_','Survived']].groupby('fareband_',\
as_index=False).mean()

fare_line_data

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fareband_</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.001, 7.55]</td>
      <td>0.141304</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(7.55, 7.854]</td>
      <td>0.298851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(7.854, 8.05]</td>
      <td>0.179245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(8.05, 10.5]</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(10.5, 14.454]</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(14.454, 21.679]</td>
      <td>0.420455</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(21.679, 27.0]</td>
      <td>0.516854</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(27.0, 39.688]</td>
      <td>0.373626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(39.688, 77.958]</td>
      <td>0.528090</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(77.958, 512.329]</td>
      <td>0.758621</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in range(fareband_count):
    left = fare_line_data.loc[i,'fareband_'].left
    left = float(left)
    right = fare_line_data.loc[i,'fareband_'].right

    test.loc[(test['fareband_num'] > left) & (test['fareband_num'] <= right),'fareband_num'] = 1 + i
```


```python
test.loc[(test['fareband_num'] > 512) ,'fareband_num'] = 10
```


```python
test[['Fare','sex']].groupby('sex').median()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.5125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.loc[test.Fare.isnull(),'Sex']
```




    152    male
    Name: Sex, dtype: object




```python
test.loc[test['Fare'].isnull(),'fareband_num'] = 4
```

手动处理test中Fare为NaN的数据，将其归到4。其为男性，男性中位数船费为13，属于 4


```python
test[['sex','fareband_num']].groupby('fareband_num',as_index=False).count()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fareband_num</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature = ['Pclass','SibSp','Parch','sex','title',
             'embarked','isalone','age*pclass','fareband_num']
```


```python
test_feature = test[feature]
```


```python
test_feature.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 9 columns):
    Pclass          418 non-null int64
    SibSp           418 non-null int64
    Parch           418 non-null int64
    sex             418 non-null object
    title           418 non-null object
    embarked        418 non-null object
    isalone         418 non-null int64
    age*pclass      418 non-null float64
    fareband_num    418 non-null float32
    dtypes: float32(1), float64(1), int64(4), object(3)
    memory usage: 27.8+ KB


数据中不能有一个空值NaN


```python
tdt = tree.DecisionTreeClassifier()
ty = model.predict(test_feature)
test['Survived'] = ty
```




```python
out = test[['PassengerId','Survived']]
out.to_csv('titanic_model_dt.csv',
           index=False)
```

## 导出数据
```
pandas.DataFrame.to_csv(
path_or_buf=None,      # 存储路径
sep=', ',
na_rep='',
float_format=None,
columns=None,
header=True,          # 是否包括标题
index=True,
index_label=None,
mode='w',
encoding=None,
compression=None,
quoting=None,
quotechar='"',
line_terminator='\n',
chunksize=None,
tupleize_cols=None,
date_format=None,
doublequote=True,
escapechar=None,
decimal='.'  )
```
然后将数据提交到[kaggle](https://www.kaggle.com/c/titanic)即完成

将清洗后的数据保存下来，再用9个模型分别尝试，找出最优的模型：Titanic_sklearn_9models.ipynb
```python
titanic[['Survived','Pclass','SibSp','Parch','sex','title',
             'embarked','isalone','age*pclass','fareband_num']].\
to_csv('titanic_new_fetr.csv',
       index=False)
```


```python
test[['Pclass','SibSp','Parch','sex','title',
             'embarked','isalone','age*pclass','fareband_num']].\
to_csv('test_new_fetr.csv',index=False)
```
