# coding = UTF-8
from . import TCGPRdata
from . import TCGPRfeature
from . import TCGPRdata_r


# =============================================================================
# Public estimators
# =============================================================================


def fit(filePath, Mission = 'DATA', Sequence = 'forward', initial_set_cap=3, sampling_cap=1, ratio=0.1, target=1,up_search = 200, exploit_coef=2,Self_call = True, exploit_model = False, alpha=1e-10, n_restarts_optimizer=10,
          normalize_y=True,  ):

    """
    Algorithm name: Tree classifier for gaussian process regression
    outliers detection, features selection

    ==================================================================
    Please feel free to open issues in the Github :
    https://github.com/Bin-Cao/TCGPR
    or 
    contact Bin Cao (bcao@shu.edu.cn)
    in case of any problems/comments/suggestions in using the code. 
    ==================================================================

    ==================================================================
    encode log: 
        March 14 2022 first version for data screening / Bin CAO
        Jun 16 2022 add note / Bin CAO
        Jan 12 2023 revise code framework / Bin CAO
        Jan 19 2023 supplement feature selection function / Bin CAO
        Feb 3 2023 debug in multi-targets / Bin CAO
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
    + for Mission = 'DATA':
    ++ if Sequence = 'forward':
        initial_set_cap : the capacity of the initial dataset
        int, default = 3, recommend = 3-10
        or a list : i.e.,  
        [3,4,8], means the 4-th, 5-th, 9-th datum will be collected as the initial dataset
    ++ elif Sequence = 'backward':
        param initial_set_cap is masked 
    + for Mission = 'FEATURE':
        initial_set_cap : the capacity of the initial featureset
        int, default = 1, recommend = 1-5
        or a list : i.e.,  
        [3,4,8], means the 4-th, 5-th, 9-th feature will be selected as the initial characterize

    :param sampling_cap: 
    + for Mission = 'DATA':
        int, the number of data added to the updating dataset at each iteration, default = 1, recommend = 1-5
    + for Mission = 'FEATURE':
        int, the number of features added to the updating feature set at each iteration, default = 1, recommend = 1-3

    :param ratio: 
    + for Mission = 'DATA':
    ++ if Sequence = 'forward':
        tolerance, lower boundary of R is (1-ratio)Rmax, default = 0.1, recommend = 0-0.3
    ++ elif Sequence = 'backward':
        tolerance, lower boundary of R is (1+ratio)R[last], default = 0.1, recommend = 0.001-0.05
    + for Mission = 'FEATURE':
        tolerance, lower boundary of R is (1+ratio)R[last], default = 0.1, recommend = 0.001-0.05

    :param target:
    used in feature selection when Mission = 'FEATURE'
        int, default 1, the number of target in regression mission
        target = 1 for single_task regression and =k for k_task regression (Multiobjective regression)
    otherwise : param target is masked 
    
    :param up_search: 
    + for Mission = 'DATA':
        up boundary of candidates for brute force search, default = 2e2 , recommend =  2e2-2e4
    + for Mission = 'FEATURE':
        up boundary of candidates for brute force search, default = 20 , recommend =  10-2e2

    :param exploit_coef: constrains to the magnitude of variance in Cal_EI function, default = 2, recommend = 2

    :param Self_call: 
    + for Mission = 'DATA':
    ++ if Sequence = 'forward':
        the calculation model of TCGPR, default = True, 
        Self_call=True, TCGPR will be executed repeatedly on the remained dataset. 
    ++ elif Sequence = 'backward': Self_call is masked
    + for Mission = 'FEATURE': Self_call is masked

    :param exploit_model: boolean, default, False
        exploit_model == True, the searching direction will be R only! GGMF will not be used!

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
    import pandas as pd
    for Mission = 'DATA':
    ++ if Sequence = 'forward':
        #coding=utf-8
        from TCGPR import TCGPR
        dataSet = "data.csv"
        initial_set_cap = 3
        sampling_cap =2
        ratio = 0.2
        up_search = 500
        TCGPR.fit(
            filePath = dataSet, initial_set_cap = initial_set_cap, sampling_cap = sampling_cap,
            ratio = ratio, up_search = up_search
                )
        note: default setting of Mission = 'DATA', No need to declare
    ++ elif Sequence = 'backward':
        #coding=utf-8
        from TCGPR import TCGPR
        dataSet = "data.csv"
        initial_set_cap = 3
        sampling_cap =2
        ratio = 0.001 # recommend a small float value
        up_search = 500
        TCGPR.fit(
            filePath = dataSet, Sequence = 'backward', sampling_cap = sampling_cap,
            ratio = ratio, up_search = up_search
                )
        note: default setting of Mission = 'DATA', No need to declare; initial_set_cap is masked 
    + for Mission = 'FEATURE': 
        #coding=utf-8
        from TCGPR import TCGPR
        dataSet = "data.csv"
        sampling_cap =2
        ratio = 0.001 # recommend a small float value
        up_search = 500
        TCGPR.fit(
            filePath = dataSet, Mission = 'FEATURE', initial_set_cap = initial_set_cap, sampling_cap = sampling_cap,
            ratio = ratio, up_search = up_search
                )
        note: for feature selection, Mission should be declared as  Mission = 'FEATURE' ! 
    
    References
    ----------
    .. [1] https://github.com/Bin-Cao/TCGPR/blob/main/Intro/TCGPR.pdf

    .. [2] Software copyright : Zhang Tong-yi, Cao Bin, Sun Sheng. 
        Tree-Classifier for Gaussian Process Regression. 
        2022SR1423038 (2022)
    
    .. [3] Patent : Zhang Tong-yi, Cao Bin, Yuan Hao, Wei Qinghua, Dong Ziqiang. 
        Tree-Classifier for Gaussian Process Regression. (一种高斯过程回归树分类器多元合金异常数据识别方法) 
        CN 115017977 A(2022)
    """
    if type(up_search) != int:
        print('Type Error: %s' % type(up_search), ' must be integer!')

    if Mission == 'DATA':
        if Sequence == 'forward':
            print('The first execution of TCGPR : Data screening ; forward version')
            TCGPRdata.cal_TCGPR(filePath=filePath, initial_set_cap=initial_set_cap, sampling_cap=sampling_cap, ratio=ratio, up_search = up_search,target = target, exploit_coef=exploit_coef,
                alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
            print('One conherenced dataset has been saved !')
            print('='*100)

            
            if Self_call == True:
                print('Self_call is True, the remianed data will be screened by TCGPR')
                Num = 0
                while True:
                    Num += 1 
                    print('The {num}-th execution of TCGPR'.format(num = Num))
                    data = pd.read_csv('Dataset remained by TCGPR.csv')
                    try:
                        if len(data.iloc[:,0]) < initial_set_cap:
                            print('The outliers have been detected ')
                            break
                    except:
                        if len(data.iloc[:,0]) < len(initial_set_cap):
                            print('The outliers have been detected ')
                            break
                    else:
                        try:
                            TCGPRdata.cal_TCGPR(filePath="Dataset remained by TCGPR.csv", initial_set_cap=initial_set_cap, sampling_cap=sampling_cap, ratio=ratio, up_search = up_search, target = target,exploit_coef=exploit_coef,
                                alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
                        except:
                            break
            else:
                pass
        elif Sequence == 'backward':
            print('Modle Data screening of backward searching is executed : param initial_set_cap is masked ')
            print('The first execution of TCGPR : Data screening; backward version')
            TCGPRdata_r.cal_TCGPR(filePath=filePath, initial_set_cap=initial_set_cap, sampling_cap=sampling_cap, ratio=ratio, up_search = up_search,target = target, exploit_coef=exploit_coef,
                alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
            print('One conherenced dataset has been saved !')
            print('='*100)

    if Mission == 'FEATURE':
        print('The first execution of TCGPR : Feature selection')
        TCGPRfeature.cal_TCGPR(filePath=filePath, initial_set_cap=initial_set_cap, sampling_cap=sampling_cap, ratio=ratio, up_search = up_search, target = target, exploit_coef=exploit_coef,
            alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, exploit_model = exploit_model)
        print('One conherenced dataset with selected features has been saved !')
        print('='*100)

        