from myHMM import GaussianHMM
import tracking
import datetime
import numpy as np
from database import StockData
from matplotlib import pyplot as plt

def backtrack_on_index(stk_name = "000300", \
                       start = datetime.date(2006,1,1), \
                       split = datetime.date(2018,1,1), \
                       end = datetime.date(2019,4,1), \
                       tstates = 5, \
                       idx_names = ["ATR_14"]):
    '''
    Backrack the performance of HMM with/without bayesian reference. Record the output as both images and txt files.
    '''
        
    track_len = 7 # track_len to judge if a state is profitable
    profit_prob = 0.7
    rate = 0.00001
    lamb = 0.1
    #fetching data from database
    stockdata = StockData()
    tdates = np.array([datetime.datetime.strptime(i, "%Y-%m-%d").date() \
              for i in stockdata.get_data(stk_name,"date",start,end).reshape(-1)])
    tclose = stockdata.get_data(stk_name,"close",start,end).reshape(-1)
    indicator = np.zeros((len(tdates),len(idx_names)))
    for i in range(len(idx_names)):
        n = idx_names[i]
        idx_value = stockdata.get_data(stk_name,n,start,end).reshape(-1)
        indicator[:,i] =idx_value
    
    split_idx = np.argmax(tdates>split)
    
    # training HMM model on the training set
    model = GaussianHMM(n_state=tstates, x_size=np.shape(indicator)[1], n_iter=50,prt = False)
    model.init_mean_cov(indicator[:split_idx,:])
    model.train(indicator[:split_idx,:])
    
    # get hidden states and last_state_probabilities
    hidden_states, gamma = model.approximate(indicator[:split_idx,:])
    l_prob = gamma[-1,:]
    if min(l_prob) != min(l_prob):
        l_prob = np.ones(model.n_state)/model.n_state
    # get rtn_series for each state
    rtn_array = np.ones((len(indicator),tstates))
    for s in range(tstates):
        signal = hidden_states == s
        rtn_series, _rtn_ = tracking.rtn_series(tclose[:split_idx], signal)
        rtn_array[:split_idx,s] = rtn_series
    
    # the testset
    test_signal = np.zeros(len(indicator)-split_idx)
    test_states = np.zeros((len(indicator)-split_idx,tstates))
    for i in range(split_idx,len(indicator)):
        # the following operations are not based on info for day[i]
        n_prob = np.dot(l_prob, model.transmat)
        profiting = rtn_array[i-1]/rtn_array[i-track_len-1]-1 > rate
        test_states[i-split_idx] = profiting
        profit_rate = np.sum(profiting*n_prob)
        if profit_rate > profit_prob:
            test_signal[i-split_idx] = 1
        # now we have info for day[i], renew l_prob and rtn_array
        n_val = indicator[i]
        l_prob, n_s = model.bayesian_refresh_new(l_prob,n_val,lamb=lamb)
        #print(l_prob)
        for s in range(tstates):
            if s == n_s:
                rtn_array[i,s] = rtn_array[i-1,s]*tclose[i]/tclose[i-1]
            else:
                rtn_array[i,s] = rtn_array[i-1,s]
                
    rtn_test, ind_info = tracking.rtn_series(tclose[split_idx:], test_signal, expense = (0,0))
    
    ind_info["stk_name"] = stk_name
    ind_info["start"] = start
    ind_info["split"] = split
    ind_info["end"] = end
    ind_info["tstates"] = tstates
    ind_info["idx_names"] = idx_names
    
    # recording data
    fig1 = plt.figure()
    plt.plot(tdates, tclose/tclose[0])
    for i in range(tstates):
        plt.plot(tdates,rtn_array[:,i])
    fig1.autofmt_xdate()
    fig1.savefig("record_lambda{4}/n({0})_lamb({4})_s({1})_tp({2})_idx({3})_fig1.png"\
                 .format(stk_name, tstates, str(start)+"+"+str(split)+"+"+str(end), str(idx_names), lamb), dpi=200)
    
    fig2 = plt.figure()
    plt.plot(tdates[split_idx:],tclose[split_idx:]/tclose[split_idx])
    plt.plot(tdates[split_idx:],rtn_test)
    fig2.autofmt_xdate()
    fig2.savefig("record_lambda{4}/n({0})_lamb({4})_s({1})_tp({2})_idx({3})_fig2.png"\
                 .format(stk_name, tstates, str(start)+"+"+str(split)+"+"+str(end), str(idx_names), lamb), dpi=200)
    
    with open("record_lambda{4}/n({0})_lamb({4})_s({1})_tp({2})_idx({3})_info.txt"\
              .format(stk_name, tstates, str(start)+"+"+str(split)+"+"+str(end), str(idx_names), lamb), 'w') as f:

        for k,v in ind_info.items():
            f.write(k+" : " + str(v) + "\n")
            
            
    return ind_info

if __name__ == "__main__":
    ind_info = backtrack_on_index(stk_name = "000300", \
                       start = datetime.date(2006,1,1), \
                       split = datetime.date(2015,1,1), \
                       end = datetime.date(2018,1,1), \
                       tstates = 5, \
                       idx_names =  ["ATR_14","ma_20","ma_diff_5_20","AR_26","volume"])