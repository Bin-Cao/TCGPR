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
        __init__.py
        OutliersIdentification.py
        DatasetPartition.py
    feature/
        __init__.py
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
CV = 10
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
CV = 10
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
CV = 10
TCGPR.fit(
    filePath = dataSet, Mission = 'FEATURE', initial_set_cap = initial_set_cap, sampling_cap = sampling_cap,
    up_search = up_search,CV=CV
        )
# note: for feature selection, Mission should be declared as Mission = 'FEATURE' ! 
```

## output : 
+ [Dataset remained by TCGPR.csv]

## About 
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 


## Contributing / 共建
Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration.
