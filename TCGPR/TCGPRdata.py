# coding = UTF-8
import time
import warnings
import copy
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut
from itertools import combinations

# the main function
def cal_TCGPR(filePath, initial_set_cap=3, sampling_cap=1, ratio=0.1, up_search = 2e2,target = 1, exploit_coef=2, alpha=1e-10, n_restarts_optimizer=10,
          normalize_y=True, exploit_model = False):

    """
    Algorithm name: Tree classifier for gaussian process regression
    23 April 2022, version 1, Bin Cao, MGI, SHU, Shanghai, CHINA.

    :param filePath: the input dataset in csv

    :param initial_set_cap: the initial dataset capacity, default = 3, recommend = 3-10
            or a list : i.e.,  
            [3,4,8], means the 4-th, 5-th, 9-th datum will be collected as the initial dataset

    :param sampling_cap: the number of data added to the updating dataset at each iteration, default = 1, recommend = 1-5

    :param ratio: tolerance, lower boundary of R is (1-ratio)Rmax, default = 0.1, recommend = 0-0.3

    :param up_search: up boundary of candidates for brute force search, default = 2e2 , recommend =  2e2-2e4

    :param exploit_coef: constrains to the magnitude of variance in Cal_EI function,default = 2, recommend = 2

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
    """


    # import dataset
    data = pd.read_csv(filePath)
    # be executed time
    timename = time.localtime(time.time())
    namey, nameM, named, nameh, namem, mames = timename.tm_year, timename.tm_mon, timename.tm_mday, timename.tm_hour, timename.tm_min, timename.tm_sec
    warnings.filterwarnings('ignore')

    fea_name = data.columns
    fea_num = len(fea_name)
    data = np.array(data)
    # global settings
    fitting_goodness_track = []
    GGMF_GLOBAL_Track = []

    if type(initial_set_cap) != int:
        if type(initial_set_cap) == list:
            zero_ini_dataset = np.zeros((len(initial_set_cap), fea_num))
            for datum in range(len(initial_set_cap)):  
                # read in the index of datum in raw dataset
                zero_ini_dataset[datum] = data[initial_set_cap[datum]]

            ini_data = copy.deepcopy(zero_ini_dataset)
            
            # evaluate the quality of initial dataset
            GGMF_ini,Rvalue,_ = dataset_eval(ini_data,target,n_restarts_optimizer,alpha,normalize_y,exploit_model)
          
            fitting_goodness_track.append(1 - Rvalue)
            GGMF_GLOBAL_Track.append(GGMF_ini)

            up_data = np.delete(data, initial_set_cap, axis=0)
            outdata = pd.DataFrame(ini_data)
            outdata.columns = fea_name
            print('THE specific Initial dataset is \n', outdata)
            print('-'*100)

    elif initial_set_cap < 3:
        print('The initial capacity of dataset must be larger than 3!')
        print('Please input a right integer value')

    elif initial_set_cap >= len(data):
        print('The initial capacity of dataset must be smaller than the original dataset!')
        print('Please input a right integer value')
    
    else:
        # seraching the initial dataset
        cal_initial_dataset_index = list(combinations(range(len(data)), initial_set_cap))
        # Determine if the searching space is too large
        if len(cal_initial_dataset_index) <= up_search:
            initial_dataset_index = copy.deepcopy(cal_initial_dataset_index)
            print('The algorithm will searching all candidates by brute force searching')
        else:
            print('Candidates at current searching apace are {totalcan} \n'' {samcan} candidates will be randomly chosen '.format(totalcan=len(cal_initial_dataset_index),
                                                                         samcan=up_search))
            initial_dataset_index = list_random_del_function(cal_initial_dataset_index, up_search)

        # local setting
        GGMF_set = []
        uncer_set = []
        Rvalue_set = []
        for set_index in range(len(initial_dataset_index)):  
            # read in the combination situations
            zero_ini_dataset = np.zeros((initial_set_cap, fea_num))
            for datum in range(initial_set_cap):  
                # read in the index of datum in raw dataset
                datum_index = initial_dataset_index[set_index][datum]
                zero_ini_dataset[datum] = data[datum_index]
            _ini_data = copy.deepcopy(zero_ini_dataset)

            # testing the quality of initial dataset
            GGMF_ini,Rvalue,y_std_set = dataset_eval(_ini_data,target,n_restarts_optimizer,alpha,normalize_y,exploit_model) 
            Rvalue_set.append(Rvalue)
            uncer_set.append(np.array(y_std_set).mean())
            # loss, the lower the better
            GGMF_set.append(GGMF_ini)

        uncer_array = np.array(uncer_set)
        GGMF_array = np.array(GGMF_set)

        GGMF_empty = np.nanmax(GGMF_array) * 1.5
        _Expected_improvement = Cal_EI(GGMF_array, uncer_array, GGMF_empty, exploit_coef)
        initial_EI = np.nanmax(_Expected_improvement, axis=0)
        EI_ini_dat_index = np.where(_Expected_improvement == initial_EI)[0][0]
        fitting_goodness_track.append(1 - Rvalue_set[EI_ini_dat_index])
        GGMF_initial = GGMF_array[EI_ini_dat_index]
        GGMF_GLOBAL_Track.append(GGMF_initial)


        _zero_ini_dataset = np.zeros((initial_set_cap, fea_num))
        dat_del_set = []
        for j in range(initial_set_cap):  
            # read in the index of datum in raw dataset
            _datum_index = initial_dataset_index[EI_ini_dat_index][j]
            _zero_ini_dataset[j] = data[_datum_index]
            # del the choosen data from the original datset
            dat_del_set.append(_datum_index)
        ini_data = copy.deepcopy(_zero_ini_dataset)
        up_data = np.delete(data, dat_del_set, axis=0)

        # JUST for output the pandas format
        outdata = pd.DataFrame(ini_data)
        outdata.columns = fea_name
        print('After searching by TCGPR, THE most suitable Initial dataset is \n', outdata)
        print('-'*100)

      
    # iteration
    # adding data into the initial dataset
    if type(sampling_cap) != int:
        print('The Sampling capacity at each iteration must be a integer ')
    elif sampling_cap < 1:
        print('The Sampling capacity at each iteration must be no samller than 1!')
        print('Please input a right integer value')
    elif sampling_cap == 1:
        print('It a greedy searching strategy, since the sampling_cap is 1')
        # adding data one by one into the initial dataset
        inter = 0
        # main loop
        while True: 
            GGMF_inter_index = inter 
            inter += 1
            
            if len(up_data) < sampling_cap:
                data_in_pd = pd.DataFrame(ini_data)
                data_in_pd.columns = fea_name
                data_in_pd_rest = pd.DataFrame(up_data)
                data_in_pd_rest.columns = fea_name
                print('Only {reddata} data are remained, the sampling_cap = {sampling_cap_} is larger than it'.format(
                        reddata=len(up_data), sampling_cap_=sampling_cap))
                print('*'*100)
                print('The dataset after data screening is : \n', data_in_pd)
                print('The changes of fitting goodness is : \n', fitting_goodness_track[:-1])
                print('The dataset after data screening has been saved successfully!')
                print('*'*100)
                data_in_pd.to_csv('Dataset selected by TCGPR {year}.{month}.{day}-{hour}.{minute}.{second}.csv'
                        .format(year=namey, month=nameM, day=named, hour=nameh, minute=namem, second=mames),index=False)
                data_in_pd_rest.to_csv('Dataset remained by TCGPR.csv',index=False)
                break

            else:
                _uncer_array,GGMF_inter_array,_Rvalue_set,_ = best_supplement(target,up_data,ini_data,n_restarts_optimizer,alpha,normalize_y,exploit_model,sampling_cap,up_search,fea_num,single_samples=True)

                Expected_improvement = Cal_EI(GGMF_inter_array, _uncer_array, GGMF_GLOBAL_Track[GGMF_inter_index],exploit_coef)
                # select the best datum with highest improvement
                EI_max = np.nanmax(Expected_improvement, axis=0)
                EI_index = np.where(Expected_improvement == EI_max)[0][0]
                fitting_goodness_track.append(1 - _Rvalue_set[EI_index])
                GGMF = GGMF_inter_array[EI_index]
                GGMF_GLOBAL_Track.append(GGMF)

                # save archive
                ulti_data = copy.deepcopy(ini_data)
                ulti_rest_data = copy.deepcopy(up_data)

                if fitting_goodness_track[inter] >= (1 - ratio) * np.array(fitting_goodness_track).max():
                    print('{iteration}-th iteration : The newly added datum is'.format(iteration=inter))
                    print(up_data[EI_index])
                    # uppdate the datset, one iteation finished
                    ini_data = np.row_stack((ini_data, up_data[EI_index]))
                    up_data = np.delete(up_data, [EI_index], axis=0)

                else:
                    data_in_pd = pd.DataFrame(ulti_data)
                    data_in_pd.columns = fea_name
                    data_in_pd_rest = pd.DataFrame(ulti_rest_data)
                    data_in_pd_rest.columns = fea_name
                    print('*'*100)
                    print('The dataset after data screening is : \n', data_in_pd)
                    print('The changes of fitting goodness is : \n', fitting_goodness_track[:-1])
                    print('The dataset after data screening has been saved successfully!')
                    print('*'*100)
                    data_in_pd.to_csv(
                        'Dataset selected by TCGPR {year}.{month}.{day}-{hour}.{minute}.{second}.csv'
                            .format(year=namey, month=nameM, day=named, hour=nameh, minute=namem, second=mames),index = False)
                    data_in_pd_rest.to_csv('Dataset remained by TCGPR.csv', index=False)
                    break

    elif sampling_cap > len(up_data):
        # forbided 
        print('The input sampling_cap is too large!' )

    elif sampling_cap > 1:
        print('The input sampling_cap is {num}'.format(num=sampling_cap))
        inter = 0
        # main loop
        while True: 
            GGMF_inter_index = inter
            inter += 1

            if len(up_data) < sampling_cap:
                data_in_pd = pd.DataFrame(ini_data)
                data_in_pd.columns = fea_name
                data_in_pd_rest = pd.DataFrame(up_data)
                data_in_pd_rest.columns = fea_name
                print('Only {reddata} data are remained, the sampling_cap = {sampling_cap_} is larger than it'.format(
                    reddata=len(up_data), sampling_cap_=sampling_cap))
                print('*'*100)
                print('The dataset after data screening is : \n', data_in_pd)
                print('The changes of fitting goodness is : \n', fitting_goodness_track[:-1])
                print('The dataset after data screening has been saved successfully!')
                print('*'*100)
                data_in_pd.to_csv('Dataset selected by TCGPR {year}.{month}.{day}-{hour}.{minute}.{second}.csv'
                        .format(year=namey, month=nameM, day=named, hour=nameh, minute=namem, second=mames),index=False)
                data_in_pd_rest.to_csv('Dataset remained by TCGPR.csv', index=False)
                break

            else:
                _uncer_array,GGMF_inter_array,_Rvalue_set,sampling_index = best_supplement(target,up_data,ini_data,n_restarts_optimizer,alpha,normalize_y,exploit_model,sampling_cap,up_search,fea_num,single_samples=0)
                Expected_improvement = Cal_EI(GGMF_inter_array, _uncer_array, GGMF_GLOBAL_Track[GGMF_inter_index],exploit_coef)
                EI_max = np.nanmax(Expected_improvement, axis=0)
                EI_index = np.where(Expected_improvement == EI_max)[0][0]
                fitting_goodness_track.append(1 - _Rvalue_set[EI_index])
                GGMF = GGMF_inter_array[EI_index]
                GGMF_GLOBAL_Track.append(GGMF)

                # save archive
                ulti_data = copy.deepcopy(ini_data)
                ulti_rest_data = copy.deepcopy(up_data)

                if fitting_goodness_track[inter] >= (1 - ratio) * np.array(fitting_goodness_track).max():
                    print('{iteration}-th iteration : The newly added data are'.format(iteration=inter))
                    del_index = []
                    for i in range(len(sampling_index[EI_index])):
                        print(up_data[sampling_index[EI_index][i]])

                        # uppdate the datset
                        ini_data = np.row_stack((ini_data, up_data[sampling_index[EI_index][i]]))
                        del_index.append(sampling_index[EI_index][i])
                    up_data = np.delete(up_data, del_index, axis=0)

                else:
                    data_in_pd = pd.DataFrame(ulti_data)
                    data_in_pd.columns = fea_name
                    data_in_pd_rest = pd.DataFrame(ulti_rest_data)
                    data_in_pd_rest.columns = fea_name
                    print('*'*100)
                    print('The dataset after data screening is : \n', data_in_pd)
                    print('The changes of fitting goodness is : \n', fitting_goodness_track[:-1])
                    print('The dataset after data screening has been saved successfully!')
                    print('*'*100)
                    data_in_pd.to_csv(
                        'Dataset selected by TCGPR {year}.{month}.{day}-{hour}.{minute}.{second}.csv'
                            .format(year=namey, month=nameM, day=named, hour=nameh, minute=namem, second=mames),
                        index=False)
                    data_in_pd_rest.to_csv('Dataset remained by TCGPR.csv', index=False)

                    break
    else:
        print('The initial capacity of dataset must be a integer or a list ')


# define the function for calculating 1-R value
def PearsonR(X, Y,target):
    X_ = copy.deepcopy(np.array(X)).reshape(-1,target)
    Y_ = copy.deepcopy(np.array(Y)).reshape(-1,target)
    R_value = 0
    for j in range(target):
        X = X_[:,j]
        Y = Y_[:,j]
        xBar = np.mean(X)
        yBar = np.mean(Y)
        SSR = 0
        varX = 0
        varY = 0
        for i in range(0, len(X)):
            diffXXBar = X[i] - xBar
            diffYYBar = Y[i] - yBar
            SSR += (diffXXBar * diffYYBar)
            varX += diffXXBar ** 2
            varY += diffYYBar ** 2
        SST = math.sqrt(varX * varY)
    R_value += (1 - SSR / SST)
    return R_value/target


# define the function for calculating the standard normal probability distribution
def norm_des(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)


# def the global gaussian messy factor
def GGMfactor(lik_hood, Rvalue, Length_scale, exploit_model):
    if exploit_model == True:
        return Rvalue
    else:
        return np.log(lik_hood + 1) * Rvalue / np.log(Length_scale + 1)

# def the  expected decrease
def Cal_EI(pre, std, currentmin, exploit_coef ):
    if len(pre) == len(std):
        std = np.log(exploit_coef + std) + 1e-10
        EI = []
        for i in range(len(pre)):
            each_EI = (currentmin - pre[i]) * norm.cdf((currentmin - pre[i]) / std[i]) \
                      + std[i] * norm_des((currentmin - pre[i]) / std[i])
            EI.append(each_EI)
        return np.array(EI)
    else:
        return 'Num Error'

# def the sampling function
def list_random_del_function(list_, up_save_num):
    del_num = int(len(list_) - up_save_num)
    if del_num <= up_save_num:
        del_lis = random.sample(range(0,len(list_)),del_num)
        out_list = [n for i, n in enumerate(list_) if i not in del_lis]
    else:
        out_list = []
        sav_index = random.sample(range(0,len(list_)),up_save_num)
        for i in range(up_save_num):
            out_list.append(list_[sav_index[i]])
    return out_list

def dataset_eval(dataset,target,n_restarts_optimizer,alpha,normalize_y,exploit_model):
    # evaluate the quality of dataset
    KErnel = RBF(length_scale_bounds = (1e-2, 1e2))
    X = dataset[:, :-target]
    Y = dataset[:, -target:]
    loo = LeaveOneOut()
    y_pre_set = []
    y_std_set = []
    ls_set = []
    neg_lik_hood_set = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = Y[train_index], Y[test_index]
        Gpr_i = GaussianProcessRegressor(kernel=KErnel,
                                        n_restarts_optimizer=n_restarts_optimizer,
                                        alpha=alpha,
                                        normalize_y=normalize_y,
                                        random_state=0).fit(X_train, y_train)

        LS_i = np.exp(Gpr_i.kernel_.theta)
        ls_set.append(LS_i)
        neg_li = - Gpr_i.log_marginal_likelihood(LS_i)
        neg_lik_hood_set.append(neg_li)
        y_pre = Gpr_i.predict(X_test, return_std=True)[0]
        y_std = Gpr_i.predict(X_test, return_std=True)[1]
        y_pre_set.append(y_pre[0])
        y_std_set.append(y_std.mean())
    Rvalue = PearsonR(y_pre_set, Y,target)
    _LS_i = np.array(ls_set).mean()
    Lik_hood_value = np.array(neg_lik_hood_set).mean()
    GGMF_value = GGMfactor(Lik_hood_value, Rvalue, _LS_i, exploit_model)
    return GGMF_value,Rvalue,y_std_set

def best_supplement(target,up_data,ini_data,n_restarts_optimizer,alpha,normalize_y,exploit_model,sampling_cap,up_search,fea_num,single_samples=True):
    # add one datum 
    if single_samples == True:
        GGMF_inter_set = []
        _uncer_set = []
        _Rvalue_set = []
        for i in range(len(up_data)):
            dataset_inter = copy.deepcopy(ini_data)
            dataset_inter = np.row_stack((dataset_inter, up_data[i]))

            GGMF_inter,_Rvalue, _y_std_set= dataset_eval(dataset_inter,target,n_restarts_optimizer,alpha,normalize_y,exploit_model)

            _Rvalue_set.append(_Rvalue)
            _uncer_set.append(np.array(_y_std_set).mean())
            GGMF_inter_set.append(GGMF_inter)

        _uncer_array = np.array(_uncer_set)
        GGMF_inter_array = np.array(GGMF_inter_set)
        sampling_index = None
    
    # add a set of data  
    else:
        GGMF_inter_set = []
        _uncer_set = []
        _Rvalue_set = []
        cal_sampling_index = list(combinations(range(len(up_data)), sampling_cap))
        # Determine if the searching space is too large
        if len(cal_sampling_index) <= up_search:
            sampling_index = copy.deepcopy(cal_sampling_index)
            print('The algorithm will searching all candidates by brute force searching')
        else:
            print(' Candidates at current searching space are {totalcan} \n'
                    ' {samcan} candidates will be randomly chosen and compared'.format(totalcan = len(cal_sampling_index), samcan= up_search))
            sampling_index = list_random_del_function(cal_sampling_index,up_search)

       
        for k in range(len(sampling_index)):  # read in the combination situations
            zero_samp_dataset = np.zeros((sampling_cap, fea_num))
            for datum in range(sampling_cap):  # read in the index of datum in raw dataset
                datum_index = sampling_index[k][datum]
                zero_samp_dataset[datum] = up_data[datum_index]
            sam_data = copy.deepcopy(zero_samp_dataset)

            dataset_inter = copy.deepcopy(ini_data)
            dataset_inter = np.row_stack((dataset_inter, sam_data))
            
            GGMF_inter,_Rvalue, _y_std_set= dataset_eval(dataset_inter,target,n_restarts_optimizer,alpha,normalize_y,exploit_model)
            _Rvalue_set.append(_Rvalue)
            _uncer_set.append(np.array(_y_std_set).mean())
            GGMF_inter_set.append(GGMF_inter)

        _uncer_array = np.array(_uncer_set)
        GGMF_inter_array = np.array(GGMF_inter_set)
    return _uncer_array,GGMF_inter_array,_Rvalue_set,sampling_index

