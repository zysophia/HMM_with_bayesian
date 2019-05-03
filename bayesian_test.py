from myHMM import GaussianHMM
import tracking
import datetime
import numpy as np
from database import StockData
from matplotlib import pyplot as plt



stk_name = "000300"
start = datetime.date(2006,1,1)
split = datetime.date(2015,8,1)
end = datetime.date(2019,4,1)
tstates = 3
idx_names = ["return_1"]#,"ma_20","volatility_20","bias_20","ma_5", "volatility_5", \
           #"bias_5","ATR_14","ma_diff_5_20","macd","RSI_6","RSI_12","BR_26",\
           #"AR_26","volume"]

stockdata = StockData()
tdates = np.array([datetime.datetime.strptime(i, "%Y-%m-%d").date() \
          for i in stockdata.get_data(stk_name,"date",start,end).reshape(-1)])
tclose = stockdata.get_data(stk_name,"close",start,end).reshape(-1)
split_idx = np.argmax(tdates>split)

recd = []

for i in range(len(idx_names)):
    indicator = np.zeros((len(tdates),1))
    n = idx_names[i]
    idx_value = stockdata.get_data(stk_name,n,start,end).reshape(-1)
    indicator[:,0] =idx_value
    
    
    model = GaussianHMM(n_state=tstates, x_size=np.shape(indicator)[1], n_iter=50,prt = False)
    model.init_mean_cov(indicator[:split_idx,:])
    model.train(indicator[:split_idx,:])
    
    hidden_states,gamma = model.approximate(indicator[:split_idx,:])
    l_prob = gamma[-1,:]
    
    print(model.transmat)
    #print(model.means)
    #print(model.covs)
    prob1 =  model.cal_Xprob(indicator[split_idx:,:])
    
    lamb = 1
    for i in range(split_idx, len(indicator)):
        new_val = indicator[i,:]
        l_prob,_ = model.bayesian_refresh_new(l_prob, new_val, lamb = 0.9)
    
    
    print(model.transmat)
    print(model.means)
    #print(model.covs)
    prob2 = model.cal_Xprob(indicator[split_idx:,:])
    
    model2 = GaussianHMM(n_state=tstates, x_size=np.shape(indicator)[1], n_iter=50,prt = False)
    model2.init_mean_cov(indicator[split_idx:,:])
    model2.train(indicator[split_idx:,:])
    
    print(model2.transmat)
    print(model2.means)
    #print(model2.covs)
    prob3 = model2.cal_Xprob(indicator[split_idx:,:])
    
    print(prob1/(len(indicator)-split_idx),prob2/(len(indicator)-split_idx),prob3/(len(indicator)-split_idx))
    recd.append([prob1,prob2,prob3])
    
recd = np.array(recd)
a = recd[:,2]-recd[:,0]
b = recd[:,2]-recd[:,1]
c = b/a
plt.plot(indicator)