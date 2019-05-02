import numpy as np
import math
from sklearn import cluster
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize

class _baseHMM():

    __metaclass__ = ABCMeta

    def __init__(self, n_state = 1,
                 startprob = None, transmat= None,
                 n_iter = 10, tol = 0.3,
                 x_size = 1,
                 prt = True):

        self.n_state = n_state
        if startprob is None:
            self.startprob = np.ones(n_state) * (1.0 / n_state)
        else:
            self.startprob = startprob
        if transmat is None:
            self.transmat = np.ones((n_state, n_state)) * (1.0 / n_state)
        else:
            self.transmat = transmat
        self.n_iter = n_iter
        self.tol = tol
        self.x_size = x_size
        self.prt = prt
    

    @abstractmethod
    def occur_prob(self, x):
        return np.array([0]*len(x))

    @abstractmethod
    def occur_prob_updated(self, X, gamma):
        pass

    @abstractmethod
    def generate_x(self, X, gamma):
        return np.array([0])


    def forward(self, X, S):

        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))

        #initialize alpha
        alpha[0] = self.occur_prob(X[0]) * self.startprob * S[0]
        # normalize
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # go on
        for i in range(1, X_length):
            alpha[i] = self.occur_prob(X[i]) * np.dot(alpha[i - 1], self.transmat) * S[i]
            c[i] = np.sum(alpha[i])
            if c[i]>0:
                alpha[i] = alpha[i] / c[i]

        return alpha, c


    def backward(self, X, S, c):

        X_length = len(X)
        # initialize beta
        beta = np.zeros((X_length, self.n_state))
        beta[-1] = np.ones((self.n_state))
        # go on
        for i in reversed(range(X_length-1)):
            beta[i] = np.dot(beta[i + 1] * self.occur_prob(X[i + 1]), self.transmat.T) * S[i]
            if c[i+1]>0:
                beta[i] = beta[i] / c[i + 1]

        return beta


    def cal_Xprob(self, X, S = None):

        X_length = len(X)
        # solve cases where we have a prior for S
        if S is None:
            S = np.ones((X_length, self.n_state))
        #get c
        _, c = self.forward(X, S)
        Xprob = np.sum(np.log(c))

        return Xprob


    def train(self, X, S = None):
        # X : np.array(lenX, features)
        # S : np.array(lenX, n_states) defalt None
        # E-M algorithms
        X_length = len(X)
        # solve cases where we have a prior for S
        if S is None:
            S = np.ones((X_length, self.n_state))

        for iter in range(self.n_iter):
            if iter > 0 and self.cal_Xprob(X,S)-prob < self.tol:
                break
            else:
                prob = self.cal_Xprob(X,S)
            if self.prt:
                print("iter:", iter, "prob:", prob)
            # E of E-M
            alpha, c = self.forward(X, S)  # P(x,z)
            beta = self.backward(X, S, c)  # P(x|z)
            #gamma and theta
            gamma = alpha * beta
            theta = np.zeros((self.n_state, self.n_state))
            for i in range(1, X_length):
                if c[i] > 0:
                    theta += (1 / c[i]) * np.outer(alpha[i - 1], beta[i] * self.occur_prob(X[i])) * self.transmat
            # M of E-M
            self.startprob = gamma[0] / np.sum(gamma[0])
            for k in range(self.n_state):
                self.transmat[k] = theta[k]/np.sum(theta[k])

            self.occur_prob_updated(X, gamma)


    def generate(self, length):
        X = np.zeros((length, self.x_size))
        S = np.zeros(length)

        for i in range(length):
            if i == 0:
                S_predict = np.random.choice(self.n_state, 1, p=self.startprob)[0]
            else:
                #print(S_predict)
                S_predict = np.random.choice(self.n_state, 1, p=self.transmat[S_predict,:])[0]
            X[i] = self.generate_x(S_predict)
            S[i] = S_predict

        return X, S


    def viterbi(self, X):

        X_length = len(X)
        state = np.zeros(X_length)

        ksai = np.zeros((X_length, self.n_state))  # t-1 state with max prob, letting St=k
        delta = np.zeros((X_length, self.n_state))  # maximun prob for 1~t-1 where St=k

        _,c=self.forward(X,np.ones((X_length, self.n_state)))


        # forward
        for i in range(X_length):
            if i == 0:
                delta[0] = self.occur_prob(X[0]) * self.startprob * (1/c[0])
            else:
                for k in range(self.n_state):
                    prob_state = self.occur_prob(X[i])[k] * self.transmat[:,k] * delta[i-1]
                    delta[i][k] = np.max(prob_state)* (1/c[i])
                    ksai[i][k] = np.argmax(prob_state)

        # backward
        state[X_length - 1] = np.argmax(delta[X_length - 1,:])
        for i in reversed(range(X_length-1)):
            state[i] = ksai[i + 1][int(state[i + 1])]

        return  state, delta


    def approximate(self, X):

        S = np.ones((len(X), self.n_state))
        alpha, c = self.forward(X, S)
        beta = self.backward(X, S, c)

        gamma = alpha*beta/(alpha*beta).sum(axis = 1).reshape(-1,1)
        approx = np.zeros(len(X))
        for i in range(len(X)):
            approx[i] = np.argmax(gamma[i])

        return approx, gamma


    def prob_refresh(self, l_prob, new_val, lamb=500):
        b = self.occur_prob(new_val).reshape(1,-1)
        l = l_prob.reshape(-1,1)
        self.transmat += np.dot(l,b)/(2*lamb)
        for i in range(self.n_state):
            self.transmat[i,:] /= sum(self.transmat[i,:])
        
        prob = np.zeros(self.n_state)
        for i in range(len(prob)):
            prob[i] = sum(l_prob*self.transmat[:,i]*self.occur_prob(new_val)[i])
        
        prob = prob/sum(prob)
        return prob, np.argmax(prob)
    
    def bayesian_refresh(self, l_prob, new_val, lamb=0.01):
        b = self.occur_prob(new_val).reshape(1,-1)
        l = l_prob.reshape(-1,1)
        
        count = 0
        while count < 30:
            #print(count)
            count += 1
            lab = np.dot(l,b)*self.transmat
            add = (np.sum(lab)*(self.transmat-1)+lab)*self.transmat**(-1)
            for i in range(self.n_state):
                add[i,:] /= max(abs(add[i,:]))
            new_transmat = self.transmat + add*lamb
        
            for j in range(self.n_state):
                d = new_transmat[j,:]
                if min(d)<=0:
                    d = d - min(d) + 0.1**20
                d = d/sum(d)
                new_transmat[j,:] = d

            p1 = np.prod(self.transmat**(self.transmat-1))*np.sum(np.dot(l,b)*self.transmat)
            p2 = np.prod(new_transmat**(self.transmat-1))*np.sum(np.dot(l,b)*new_transmat)
            diff = (p2-p1)/p1  
            if diff<0.05:
                break
            self.transmat = new_transmat
            
        prob = np.zeros(self.n_state)
        for i in range(len(prob)):
            prob[i] = sum(l_prob*self.transmat[:,i]*self.occur_prob(new_val)[i])
        prob = prob/sum(prob)
        return prob, np.argmax(prob)
    
    def bayesian_refresh_new(self, l_prob, new_val, lamb=0.3):
        b = self.occur_prob(new_val).reshape(1,-1)
        l = l_prob.reshape(-1,1)
        
        count = 0
        while count < 10:
            #print(count)
            count += 1
            lab = np.dot(l,b)*self.transmat
            new_transmat = np.zeros((self.n_state,self.n_state))
            
            for i in range(self.n_state):
                for j in range(self.n_state):
                    new = (1-self.transmat[i,j])/(1-1/self.n_state-(l[i,0]*np.sum(b)/self.n_state-l[i,0]*b[0,j])/np.sum(lab))
                    #print(new)
                    if new <= 0:
                        new_transmat[i,:] = self.transmat[i,:]/(sum(self.transmat[i,:])-self.transmat[i,j])-np.exp(-10)/(self.n_state-1)
                        new_transmat[i,j] = np.exp(-10)
                    else:
                        new_transmat[i,j] = new
                new_transmat[i,:] /= sum(new_transmat[i,:])
            #print(model.transmat)
            #print(new_transmat)
            if np.sum(np.abs(new_transmat - self.transmat))<np.exp(-5):
                break
            self.transmat = (1-lamb)*self.transmat + lamb*new_transmat
            
        prob = np.zeros(self.n_state)
        for i in range(len(prob)):
            prob[i] = sum(l_prob*self.transmat[:,i]*self.occur_prob(new_val)[i])
        prob = prob/sum(prob)
        return prob, np.argmax(prob)
        
    
    

class GaussianHMM(_baseHMM):

    def __init__(self, n_state=1, x_size=1, n_iter=20, means = None, covs = None, prt = True):
        #print(startprob)
        _baseHMM.__init__(self, n_state=n_state, x_size=x_size,
                         n_iter=n_iter,prt = prt)
        self.means = means
        self.covs = covs

    def init_mean_cov(self, X):

        if self.means is None:
            mean_kmeans = cluster.KMeans(n_clusters=self.n_state)
            mean_kmeans.fit(X)
            self.means = mean_kmeans.cluster_centers_

        if self.covs is None:
            self.covs = np.zeros((self.n_state, self.x_size, self.x_size))
            for i in range(self.n_state):
                self.covs[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))


    def gaussian_prob(self, mean, cov, x):

        #print(mean,cov,x)
        z = -np.dot(np.dot((x-mean).T, np.linalg.inv(cov)),(x-mean))/2.0
        temp = pow(np.sqrt(2.0*math.pi), len(x)) * np.sqrt(np.linalg.det(cov))
        return (1.0/temp)*np.exp(z)


    def occur_prob(self, x):

        prob = np.zeros((self.n_state))
        for i in range(self.n_state):
            prob[i] = self.gaussian_prob(self.means[i],self.covs[i],x)
        return prob


    def occur_prob_updated(self, X, gamma):

        for k in range(self.n_state):
            for j in range(self.x_size):
                self.means[k][j] = np.sum(gamma[:,k] *X[:,j]) / np.sum(gamma[:,k])

            X_cov = np.dot((X-self.means[k]).T, (gamma[:,k]*(X-self.means[k]).T).T)
            self.covs[k] = X_cov / np.sum(gamma[:,k])
            if np.linalg.det(self.covs[k]) == 0:
                self.covs[k] = self.covs[k] + 0.01*np.eye(len(X[0]))

    def generate_x(self, s):
        return np.random.multivariate_normal(self.means[s],self.covs[s],1)
