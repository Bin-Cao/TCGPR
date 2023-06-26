
<h1 align="center">
  <a href=""><img src="https://github.com/Bin-Cao/TCGPR/assets/86995074/b24118cc-8915-4dd6-8718-2b184ebdf273" alt="TCGPR" width="150"></a>
  <br>
  PyTcgpr
  <br>
</h1>


🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks !

Source code : [![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/PyTcgpr/)
# TCGPR package 

Tree-Classifier for Gaussian process regression 

Cite : 
+ (Software copyright) Zhang Tong-yi, Cao Bin, Sun Sheng. Tree-Classifier for Gaussian Process Regression. 2022SR1423038 (2022), GitHub : github.com/Bin-Cao/TCGPR.


Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Algorithm Intro / 算法介绍
See [Introduction](https://github.com/Bin-Cao/TCGPR/blob/main/Intro/TCGPR.pdf)

## Installing / 安装
    pip install PyTcgpr 

## Checking / 查看
    pip show PyTcgpr 
    
## Updating / 更新
    pip install --upgrade PyTcgpr
## Structure / 结构

``` javascript
PyTcgpr/
    __init__.py
    data/
        OutliersIdentification.py
        DatasetPartition.py
    feature/
        FeaturesSelection.py
    TCGPR.py
``` 
## Running / 运行

+ Data Screening module | Partition
``` javascript
from PyTcgpr import TCGPR
dataSet = "data.csv"
initial_set_cap = 3
sampling_cap =2
up_search = 500
CV = 'LOOCV'
Task = 'Partition'
TCGPR.fit(
    filePath = dataSet, initial_set_cap = initial_set_cap,Task=Task, sampling_cap = sampling_cap,
    up_search = up_search, CV=CV
        )
# note: default setting of Mission = 'DATA', No need to declare
```
+ Data Screening module | Identification
``` javascript
from PyTcgpr import TCGPR
dataSet = "data.csv"
sampling_cap =2
up_search = 500
Task = 'Identification'
CV = 'LOOCV'
TCGPR.fit(
    filePath = dataSet, Task = Task, sampling_cap = sampling_cap,
    up_search = up_search,CV=CV
        )
# note: default setting of Mission = 'DATA', No need to declare; initial_set_cap is masked 
```
+ Feature Selection module
``` javascript
from PyTcgpr import TCGPR
dataSet = "data.csv"
sampling_cap =2
Mission = 'FEATURE'
up_search = 500
CV = 'LOOCV'
TCGPR.fit(
    filePath = dataSet, Mission = 'FEATURE', initial_set_cap = initial_set_cap, sampling_cap = sampling_cap,
    up_search = up_search,CV=CV
        )
# note: for feature selection, Mission should be declared as Mission = 'FEATURE' ! 
```

## Parameters / 参数
``` javascript
    :param Mission : str, the mission of TCGPR, 
        default Mission = 'DATA' for data screening. 
        Mission = 'FEATURE' for feature selection.

    :param filePath: the input dataset in csv format

    :param initial_set_cap: 
    for Mission = 'DATA':
        if Task = 'Partition':
        initial_set_cap : the capacity of the initial dataset
        int, default = 3, recommend = 3-10
        or a list : 
        i.e.,  
        [3,4,8], means the 4-th, 5-th, 9-th datum will be collected as the initial dataset
        elif Task = 'Identification':
        param initial_set_cap is masked 
    for Mission = 'FEATURE':
        initial_set_cap : the capacity of the initial featureset
        int, default = 1, recommend = 1-5
        or a list : i.e.,  
        [3,4,8], means the 4-th, 5-th, 9-th feature will be selected as the initial characterize

    :param sampling_cap: 
    for Mission = 'DATA':
        int, the number of data added to the updating dataset at each iteration, default = 1, recommend = 1-5
    for Mission = 'FEATURE':
        int, the number of features added to the updating feature set at each iteration, default = 1, recommend = 1-3

    :param ratio: 
    for Mission = 'DATA':
        if Task = 'Partition':
        tolerance, lower boundary of R is (1-ratio)Rmax, default = 0.2, recommend = 0~0.3
        elif Task = 'Identification':
        tolerance, lower boundary of R is (1+ratio)R[last], default = 0.0, recommend = -0.01~0.01
    for Mission = 'FEATURE':
        tolerance, lower boundary of R is (1+ratio)R[last], default = -0.01, recommend = -0.01~0.01

    :param target:
    used in feature selection when Mission = 'FEATURE'
        int, default 1, the number of target in regression mission
        target = 1 for single_task regression and =k for k_task regression (Multiobjective regression)
    otherwise : param target is masked 
    
    :param weight
    a weight imposed on R value in calculating GGMF, default = 2 , recommend =  1-10
    i.e.,
        weight * (1-R) +  mean / std (mean, std is the mean and standard deviation of length scales) 

    :param up_search: 
    for Mission = 'DATA':
        up boundary of candidates for brute force search, default = 5e2 , recommend =  2e2-2e3
    for Mission = 'FEATURE':
        up boundary of candidates for brute force search, default = 10 , recommend = 10-20

    :param exploit_coef: constrains to the magnitude of variance in Cal_EI function, default = 2, recommend = 2

    :param exploit_model: boolean, default, False
        exploit_model == True, the searching direction will be R only! GGMF will not be used!

    :param CV: cross validation, default = 10
        e.g. (int) CV = 5,10,... or str CV = 'LOOCV' for leave one out cross validation
```

## output : 
+ [Dataset remained by TCGPR.csv]

## About 
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 


## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.
