#%%

import sys
sys.path.append('/Users/xyguo/Documents/phase_trans/src')
import numpy as np

from utils.util import read, save, log_marginal_likelihood, ConvertE, predict, custom_fig, save_fig
from utils.kernels_non_stationary import ManifoldKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
import pandas as pd
import matplotlib.pyplot as plt
# kernel = read('/Users/xyguo/Documents/phase_trans/src/bic_search/out/layer_2_256.pkl')
from sklearn.preprocessing import MinMaxScaler


train_set = read( f'/Users/xyguo/Documents/phase_trans/data/processed/train_256.pkl')
test_set = read( f'/Users/xyguo/Documents/phase_trans/data/processed/test_256.pkl')

full_set = pd.concat([train_set,test_set])
full_set = full_set[full_set['n']==2]
full_set = full_set.drop(columns=['n'])
# scaler = MinMaxScaler()
# full_set.iloc[:,:-1] = scaler.fit_transform(full_set.iloc[:,:-1])





#%%
train_set = full_set[full_set['tau_q']<=256]
test_set = full_set[full_set['tau_q']>256]

x_train = train_set.iloc[:,:-1].to_numpy()
y_train = train_set.iloc[:,-1].to_numpy()
y_train = np.abs(y_train) 



print(min(y_train))
# y_train = np.log1p(y_train)

x_test = test_set.iloc[:, :-1].to_numpy()
y_test = test_set.iloc[:, -1].to_numpy()

y_test = np.abs(y_test) 
print(min(y_test))



kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(np.ones(6)), architecture=((1,2,3),(1,2,3)),
                               transfer_fct="tanh", max_nn_weight=3)

gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                              n_restarts_optimizer=0)
gp.fit(x_train, y_train)
# y_test = np.log1p(y_test)
y_pred_test = gp.predict(x_test)
y_pred_train =gp.predict(x_train)

# %%
res_dir = '/Users/xyguo/Documents/phase_trans/src/bic_search'
train_set['phi_hat'] = y_pred_train
test_set['phi_hat'] = y_pred_test
# train_set['phi_hat'] = y_pred_train
# test_set['phi_hat'] = y_pred_test

full_set = pd.concat([train_set,test_set])

for tau_q in [64.0, 128.0]:
    df = full_set[full_set['tau_q']==tau_q]
    plt.plot(df['t'],df['phi_hat'])
    plt.scatter(df['t'],df['phi'].abs())
custom_fig(
    "$t$",
    "$\phi$",
    bbox_to_anchor=None
)
save_fig(
    f'{res_dir}/fig/layer_2_256.pdf'
)

# %%

# %%
full_set
# %%
test_set
# %%
