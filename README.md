
Source code : [![](https://img.shields.io/badge/PyPI-caobin-blue)](https://pypi.org/project/TCGPR/)
# TCGPR package 
log

## TCGPR, Version 1, April, 2022.

Tree-Classifier for Gaussian process regression (TCGPR) is a data preprocessing algorithm developed for identifying outliers and/or cohesive data. TCGPR identifies outliers via Sequential Forward Identification (SFI). The SFI starts from few cohesive data, identifies outliers, which maximizes the expected decrease (ED) of the global Gaussian massy factor (GGMF) with a preset criterion of fitting-goodness, by adding a batch of p≥1 data in each sequential through the raw dataset, called an epoch. After an epoch, raw data is divided into one cohesive subset and a rest subset. In the following epoch, the rest subset processed by TCGPR is divided into cohesive and rest subsets again. The preprocessing is going on until the raw dataset is divided into a series of highly cohesive subsets and a final rest subset containing outliers only. 

Cite : 
+ (Software copyright) Zhang Tong-yi, Cao Bin, Sun Sheng. Tree-Classifier for Gaussian Process Regression. 2022SR1423038 (2022)


Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Installing / 安装
    pip install TCGPR 

## Updating / 更新
    pip install --upgrade TCGPR

## Running / 运行
### Ref. link

``` javascript
#coding=utf-8
import TCGPR
dataSet = "data1.csv"
initial_set_cap = 3
sampling_cap =2
ratio = 0.2
up_search = 500

TCGPR.fit(filePath = dataSet, initial_set_cap = initial_set_cap, sampling_cap = sampling_cap, ratio = ratio, up_search = up_search)
```

### note
``` javascript
:param filePath: the input dataset in csv

:param initial_set_cap: the initial dataset capacity, default = 3, recommend = 3-10
        or a list : i.e.,  
        [3,4,8], means the 4-th, 5-th, 9-th datum will be collected as the initial dataset

:param sampling_cap: the number of data added to the updating dataset at each iteration, default = 1, recommend = 1-5

:param ratio: tolerance, lower boundary of R is (1-ratio)Rmax, default = 0.1, recommend = 0-0.3

:param up_search: up boundary of candidates for brute force search, default = 2e2 , recommend =  2e2-2e4

:param exploit_coef: constrains to the magnitude of variance in Cal_EI function,default = 2, recommend = 2

:param Self_call: the calculation model of TCGPR, default = True, if Self_call=True, TCGPR will be executed repeatedly on the remained dataset. 

:param defined in Gpr
alpha : float or array-like of shape (n_samples), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

        def optimizer(obj_func, initial_theta, bounds):
            # * 'obj_func' is the objective function to be minimized, which
            #   takes the hyperparameters theta as parameter and an
            #   optional flag eval_gradient, which determines if the
            #   gradient is returned additionally to the function value
            # * 'initial_theta': the initial value for theta, which can be
            #   used by local optimizers
            # * 'bounds': the bounds on the values of theta
            ....
            # Returned are the best found hyperparameters theta and
            # the corresponding value of the target function.
            return theta_opt, func_min

        Per default, the 'L-BGFS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

n_restarts_optimizer : int, default=10
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, the mean and variance of
        the target values are set equal to 0 and 1 respectively. This is
        recommended for cases where zero-mean, unit-variance priors are used.
        Note that, in this implementation, the normalisation is reversed
        before the GP predictions are reported.

:param exploit_model: boolean,(default: False)
        if exploit_model == True, the searching direction will be R only! GGMF will not be used!

:return: two datasets in form of .csv 
```

output : 
+ Dataset remained by TCGPR.csv contains the outliers

## Update log / 日志
TCLR V1.0 Jan, 2023. 
*official release version*

## About 
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

