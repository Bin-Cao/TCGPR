
# TCGPR: Targeted Correlation Guided Pattern Recognition

🤝🤝🤝 **Please star ⭐️ this project to support open source efforts!** 🌍 Thank you!

---

## Citation
If you use this code in your research, please cite the following:

+ **Processing (paper):** Tianliang Li&, Bin Cao&, Tianhao Su&, ..., Lingyan Feng*, Tongyi Zhang*. (*Under review*)
+ **Patent:** Zhang Tongyi, Cao Bin, Yuan Hao, Wei Qinghua, Dong Ziqiang. Authorized Chinese Patent.

---

## Source Code

[![PyPI - TCGPR](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/PyTcgpr/)

This Python-based library is compatible with Windows, Linux, and macOS operating systems.

---

## Patent

![Patent Image](https://github.com/user-attachments/assets/32c40073-8a87-4c21-a178-15b2d51835f7)

---

## Algorithm Introduction

For detailed algorithm information, refer to the [Introduction](https://github.com/Bin-Cao/TCGPR/blob/main/Intro/TCGPR.pdf).

---

## Installation

To install TCGPR, use pip:

```bash
pip install PyTcgpr
```

---

## Checking Installation

You can check the installation details with:

```bash
pip show PyTcgpr
```

---

## Updating the Package

Update TCGPR to the latest version using:

```bash
pip install --upgrade PyTcgpr
```

---

## Running the Algorithm

### 1. **Data Screening | Partition**

```python
from PyTcgpr import TCGPR
dataSet = "data.csv"
initial_set_cap = 3
sampling_cap = 2
up_search = 500
CV = 'LOOCV'
Task = 'Partition'

TCGPR.fit(
    filePath = dataSet, 
    initial_set_cap = initial_set_cap, 
    Task = Task, 
    sampling_cap = sampling_cap,
    up_search = up_search, 
    CV = CV
)
# Note: Mission is set to 'DATA' by default. No need to declare it explicitly.
```

### 2. **Data Screening | Identification**

```python
from PyTcgpr import TCGPR
dataSet = "data.csv"
sampling_cap = 2
up_search = 500
Task = 'Identification'
CV = 'LOOCV'

TCGPR.fit(
    filePath = dataSet, 
    Task = Task, 
    sampling_cap = sampling_cap,
    up_search = up_search, 
    CV = CV
)
# Note: 'Mission' is 'DATA' by default; no need to declare it. 'initial_set_cap' is masked in this case.
```

### 3. **Feature Selection Module**

```python
from PyTcgpr import TCGPR
dataSet = "data.csv"
sampling_cap = 2
Mission = 'FEATURE'
up_search = 500
CV = 'LOOCV'

TCGPR.fit(
    filePath = dataSet, 
    Mission = Mission, 
    sampling_cap = sampling_cap,
    up_search = up_search, 
    CV = CV
)
# Note: For feature selection, 'Mission' must be explicitly set to 'FEATURE'.
```

---

## Parameters

```python
:param Mission: str, default='DATA'
    The task to perform:
    - 'DATA' for data screening
    - 'FEATURE' for feature selection

:param filePath: str
    Path to the input dataset in CSV format.

:param initial_set_cap: int or list
    Initial set capacity. For 'Partition' under 'DATA', defaults to 3.
    Can also be a list specifying the indices of the initial set.

:param sampling_cap: int, default=1
    Number of data points or features added at each iteration.

:param measure: str, default='Pearson'
    Correlation criteria. Can be 'Pearson' (R values) or 'Determination' (R² values).

:param ratio: float
    Tolerance ratio for correlation. Varies based on the mission and task.

:param target: int, default=1
    Used in feature selection. Specifies the number of targets in regression tasks.

:param weight: float, default=0.2
    Weight factor for calculating the GGMF score.

:param up_search: int, default=500
    Upper boundary for brute-force search.

:param exploit_coef: float, default=2
    Constraint on the variance in the Cal_EI function.

:param exploit_model: bool, default=False
    If True, only R values will be used for the search (GGMF will not be considered).

:param CV: int or str, default=10
    Cross-validation setting. Can be an integer (e.g., 5, 10) or 'LOOCV' for leave-one-out cross-validation.
```

---

## Output

The algorithm will output a CSV file containing the processed dataset: `Dataset_remained_by_TCGPR.csv`.

---

## Maintainers

This project is maintained by **Bin Cao**. If you encounter any issues or have suggestions, feel free to open an issue on GitHub or contact:

- **Email:** bcao686@connect.hkust-gz.edu.cn

---

## Contributing

We welcome contributions and suggestions! You can submit issues for questions, bugs, and feature requests, or submit a pull request directly. We are also open to research collaborations—please get in touch if you're interested!
