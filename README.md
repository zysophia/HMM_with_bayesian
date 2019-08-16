# HMM_with_bayesian
Hidden Markov Model is a common ML method for time series data, with strong mathematical background. Researchers have already applied bayesian reference to optimize the estimation of parameters, but there's still space for improvement. We conducted an iterative bayesian reference on HMM to further optimize the algorithm and exploit the *timeliness* of time series data. 

**The iterative bayesian reference of HMM** has enabled us to give weight to each data point with only one extra parameter. This method is more reliable than the original HMM, while much more feasible than traditional bayesian reference. Under this optimization, our Hidden Markov Model can iteratively better its performance as the data set expands. 

Stock data is a typical kind of time series data where HMM is commonly applied. To better see the performance of iterative bayesian reference, we designed a corresponding **timing strategy** and analyzed its backtesting results with/without the optimization. Backtesting results have verified the effectiveness of iterative bayesian reference on HMM.