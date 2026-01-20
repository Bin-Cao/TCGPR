
<table>
  <tr>
    <td width="160" align="center" valign="top">
      <img src="https://github.com/user-attachments/assets/7e77bd5a-42d6-45db-b8e6-2c82cac81b9d" width="140" style="border-radius: 50%;"/>
    </td>
    <td valign="top">
      <b>For any inquiries or assistance, feel free to contact Mr. CAO Bin at:</b><br>
      📧 Email: <a href="mailto:bcao686@connect.hkust-gz.edu.cn">bcao686@connect.hkust-gz.edu.cn</a><br><br>
      Cao Bin is a PhD candidate at the <b>Hong Kong University of Science and Technology (Guangzhou)</b>, 
      under the supervision of Professor <a href="https://gbaaa.org.hk/en-us/article/67">Zhang Tong-Yi</a>. His research focuses on 
      <b>AI for science</b>, especially intelligent crystal-structure analysis and discovery. 
      Learn more about his work on his 
      <a href="https://www.caobin.asia/">homepage</a>.
    </td>
  </tr>
</table>

---

# TCGPR

A Python library for divide-and-conquer (TCGPR) - an efficient strategy tailored for small datasets in materials science and beyond.

---



## Project History

* **2022**:
  TCGPR was first proposed and implemented, in collaboration with Mr. Hao Yuan (experiments cooperator) and Mr. Qinghua Wei (experiments cooperator). It was successfully applied to the optimization of lead-free solder alloys.
  → Published in *npj Computational Materials*
  [News link](https://mgi.shu.edu.cn/info/1063/3985.htm)

* **2024**:
  After two years of development, TCGPR was enhanced with sequential feature selection and outlier detection. In collaboration with Mr. Tianliang Li (experiments cooperator) and Mr. Tianhao Su (computations cooperator), it was applied to anti-tumor ferroptosis studies.
  → Published in *SMALL*
  [News link](https://www.shu.edu.cn/info/1055/363655.htm)

---

## Algorithm Overview

For an in-depth explanation of the algorithm, see the [TCGPR Introduction PDF](https://github.com/Bin-Cao/TCGPR/blob/main/Intro/TCGPR.pdf).

---

## Installation

Install TCGPR via PyPI:

```bash
pip install PyTcgpr
```

To verify the installation:

```bash
pip show PyTcgpr
```

To upgrade to the latest version:

```bash
pip install --upgrade PyTcgpr
```

---

## Getting Started

### 1. Data Screening | Partition Mode

```python
from PyTcgpr import TCGPR

TCGPR.fit(
    filePath = "data.csv",
    initial_set_cap = 3,
    sampling_cap = 2,
    up_search = 500,
    CV = 'LOOCV',
    Task = 'Partition'
)
```

### 2. Data Screening | Identification Mode

```python
from PyTcgpr import TCGPR

TCGPR.fit(
    filePath = "data.csv",
    sampling_cap = 2,
    up_search = 500,
    CV = 'LOOCV',
    Task = 'Identification'
)
```

### 3. Feature Selection Mode

```python
from PyTcgpr import TCGPR

TCGPR.fit(
    filePath = "data.csv",
    Mission = 'FEATURE',
    sampling_cap = 2,
    up_search = 500,
    CV = 'LOOCV'
)
```

---

## Main Parameters

```python
:param Mission: str, default='DATA'
    - 'DATA': Perform data screening
    - 'FEATURE': Perform feature selection

:param filePath: str
    Path to input dataset in CSV format

:param initial_set_cap: int or list
    Initial subset size or index list for Partition mode

:param sampling_cap: int, default=1
    Number of items selected per iteration

:param measure: str, default='Pearson'
    Correlation type: 'Pearson' or 'Determination'

:param ratio: float
    Tolerance threshold for correlation-based filtering

:param target: int, default=1
    Number of targets in regression (for feature selection)

:param weight: float, default=0.2
    Weight coefficient in GGMF score calculation

:param up_search: int, default=500
    Upper limit for search iterations

:param exploit_coef: float, default=2
    Variance constraint for EI acquisition function

:param exploit_model: bool, default=False
    If True, disables GGMF and uses only R values

:param CV: int or str, default=10
    Cross-validation: integer (e.g., 5, 10) or 'LOOCV'
```

---

## 📤 Output

After running, TCGPR outputs a CSV file with the remaining samples:

```bash
Dataset_remained_by_TCGPR.csv
```

---

## Source Code

[![PyPI - TCGPR](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/PyTcgpr/)

Compatible with **Windows**, **Linux**, and **macOS**.

---

## Patent

<img src="https://github.com/user-attachments/assets/32c40073-8a87-4c21-a178-15b2d51835f7" alt="Patent Image" width="400" />

---

## Developer

Maintained by **Bin Cao**
**Email**: [bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn)
Feel free to open an issue or contact me for any questions, bugs, or collaboration opportunities.

---

## Contributing

Contributions and suggestions are welcome!

* Report bugs or request features via [GitHub Issues](https://github.com/Bin-Cao/TCGPR/issues)
* Submit a pull request with improvements or fixes
* Interested in research collaboration? Please get in touch!

---

## Citation

If you use this code in your research, please cite the following papers:

* **Li T., Cao B., Su T., ... Feng L., Zhang T.**
  *Machine Learning-Engineered Nanozyme System for Synergistic Anti-Tumor Ferroptosis/Apoptosis Therapy*, **SMALL**
  [Link to paper](https://onlinelibrary.wiley.com/doi/10.1002/smll.202408750)

* **Wei Q., Cao B., Yuan H., ... Dong Z., Zhang T.**
  *Divide and conquer: Machine learning accelerated design of lead-free solder alloys with high strength and high ductility*, **npj Computational Materials**
  [Link to paper](https://www.nature.com/articles/s41524-023-01150-0)

