# TCGPR package 
TCGPR, Version 1, April, 2022. 


Tree-Classifier for Gaussian process regression (TCGPR) is a data preprocessing algorithm developed for identifying outliers and/or cohesive data. TCGPR identifies outliers via Sequential Forward Identification (SFI). The SFI starts from few cohesive data, identifies outliers, which maximizes the expected decrease (ED) of the global Gaussian massy factor (GGMF) with a preset criterion of fitting-goodness, by adding a batch of p≥1 data in each sequential through the raw dataset, called an epoch. After an epoch, raw data is divided into one cohesive subset and a rest subset. In the following epoch, the rest subset processed by TCGPR is divided into cohesive and rest subsets again. The preprocessing is going on until the raw dataset is divided into a series of highly cohesive subsets and a final rest subset containing outliers only. 

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## About 
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

