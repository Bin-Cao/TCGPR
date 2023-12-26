# mian function of TCGPR algorithm
# Author: Bin CAO <binjacobcao@gmail.com>
# coding = UTF-8

from .data.DatasetPartition import DataDP
from .data.OutliersIdentification import DataOutI
from .feature.FeaturesSelection import Feature

# =============================================================================
# Public estimators
# =============================================================================


def fit(filePath, Mission = 'DATA', Task = 'Partition', initial_set_cap=3, sampling_cap=1, measure = 'Pearson',
    ratio=None, target=1, weight = .2, up_search = None, exploit_coef=2, exploit_model = False, CV =5,
    alpha=1e-10, n_restarts_optimizer=10, normalize_y=True,):

    """
    Algorithm name: Tree classifier for gaussian process regression
    Tasks : Dataset Partition & Outliers Identification & Features Selection
    Author: Bin CAO <binjacobcao@gmail.com> 
    Guangzhou Municipal Key Laboratory of Materials Informatics, Advanced Materials Thrust,
    Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511400, Guangdong, China

    ==================================================================
    Please feel free to open issues in the Github :
    https://github.com/Bin-Cao/TCGPR
    or 
    contact Mr.Bin Cao (bcao@shu.edu.cn)
    in case of any problems/comments/suggestions in using the code. 
    ==================================================================

    Parameters
    ----------
    :param defined in TCGPR
    ==================================================================
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

    :param measure :Correlation criteria, default 'Pearson' means R values are used
        or measure = 'Determination' means R^2 values are used

    :param ratio: 
    for Mission = 'DATA':
        if Task = 'Partition':
        tolerance, lower boundary of R is (1-ratio)Rmax, default = 0.5, recommend = 0~1
        elif Task = 'Identification':
        tolerance, lower boundary of R is (1+ratio)R[last], default = 0.0, recommend = -0.1~0.1
    for Mission = 'FEATURE':
        tolerance, lower boundary of R is (1+ratio)R[last], default = -0.1, recommend = -0.1~0.1

    :param target:
    used in feature selection when Mission = 'FEATURE'
        int, default 1, the number of target in regression mission
        target = 1 for single_task regression and =k for k_task regression (Multiobjective regression)
    otherwise : param target is masked 
    
    :param weight
    a weight imposed on R value in calculating GGMF, default = .2 , recommend =  .1-1
    i.e.,
        weight * (1-R) +  mean / std (mean, std is the mean and standard deviation of length scales) 
    if weight = 0, TCGPR becomes a unsupervised algorithm, however, the cutoff threshold is still related to R value

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

    :param defined in Gpr of sklearn package
    ==================================================================
    [sklearn]alpha : float or array-like of shape (n_samples), default=1e-10
            Value added to the diagonal of the kernel matrix during fitting.
            Larger values correspond to increased noise level in the observations.
            This can also prevent a potential numerical issue during fitting, by
            ensuring that the calculated values form a positive definite matrix.
            If an array is passed, it must have the same number of entries as the
            data used for fitting and is used as datapoint-dependent noise level.
            Note that this is equivalent to adding a WhiteKernel with c=alpha.
            Allowing to specify the noise level directly as a parameter is mainly
            for convenience and for consistency with Ridge.

    [sklearn]optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
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

    [sklearn]n_restarts_optimizer : int, default=10
            The number of restarts of the optimizer for finding the kernel's
            parameters which maximize the log-marginal likelihood. The first run
            of the optimizer is performed from the kernel's initial parameters,
            the remaining ones (if any) from thetas sampled log-uniform randomly
            from the space of allowed theta-values. If greater than 0, all bounds
            must be finite. Note that n_restarts_optimizer == 0 implies that one
            run is performed.

    [sklearn]normalize_y : boolean, optional (default: False)
            Whether the target values y are normalized, the mean and variance of
            the target values are set equal to 0 and 1 respectively. This is
            recommended for cases where zero-mean, unit-variance priors are used.
            Note that, in this implementation, the normalisation is reversed
            before the GP predictions are reported.

    :return: datasets

    Examples
    --------
    Data Screening module
    Partition & Identification
    --------
    for Mission = 'DATA':
        #coding=utf-8
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
        note: default setting of Mission = 'DATA', No need to declare
        
        
        #coding=utf-8
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
        note: default setting of Mission = 'DATA', No need to declare; initial_set_cap is masked 
    --------
    Feature Selection module
    --------
        #coding=utf-8
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
        note: for feature selection, Mission should be declared as Mission = 'FEATURE' ! 
    
    References
    ----------
    .. [1] https://github.com/Bin-Cao/TCGPR/blob/main/Intro/TCGPR.pdf
    """
    
    if type(up_search) != int and up_search != None:
        print('Type Error, Param up_search must be an integer!')
    
    if CV == 'LOOCV':
        print('Leave one out cross validation is applied in TCGPR')
    elif type(CV) == int :
        print('The {} folds cross validation is applied in TCGPR'.format(CV))
    else: 
        print('Type Error: %s' % type(CV) ,'must be integer of str')
        print('E.g., CV = 10 or CV = \'LOOCV\'')

    if type(initial_set_cap) == int:
        initial_cap = initial_set_cap
    elif type(initial_set_cap) == list: 
        initial_cap = len(initial_set_cap)
    
    if Mission == 'DATA':
        if up_search == None:
            up_search = 500
        if Task == 'Partition':
            print('Execution of TCGPR : Dataset Partition Module')
            if ratio == None:
                ratio = 0.5
            # In this module, the value of cv need to be detected in advance 
            if CV == 'LOOCV':
                pass
            elif CV > initial_cap:
                print('The value of initial_set_cap must larger than the value of CV !')

            DataDP(filePath=filePath, initial_set_cap=initial_set_cap, sampling_cap=sampling_cap,measure=measure, ratio=ratio, up_search = up_search,target = target, weight=weight,exploit_coef=exploit_coef,
                CV=CV,alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
            print('The conherenced dataset has been saved !')
            print('='*100)

        elif Task == 'Identification':
            print('Execution of TCGPR : Outlier Identification Module | param initial_set_cap is masked')
            if ratio == None:
                ratio = 0.0
            DataOutI(filePath=filePath, sampling_cap=sampling_cap, measure=measure,ratio=ratio, up_search = up_search,target = target, weight=weight,exploit_coef=exploit_coef,
                CV=CV,alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
            print('The Outliers has been deleted !')
            print('='*100)

    if Mission == 'FEATURE':
        if up_search == None:
            up_search = 10
        if ratio == None:
            ratio = -0.1
        print('Execution of TCGPR : Feature Selection Module')
        Feature(filePath=filePath, initial_set_cap=initial_set_cap, sampling_cap=sampling_cap, measure=measure,ratio=ratio, up_search = up_search, target = target,weight=weight, exploit_coef=exploit_coef,
            CV=CV,alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
        print('Important features has been saved !')
        print('='*100)

        