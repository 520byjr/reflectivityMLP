#opt/anaconda3/bin/python3.7
"""
    compute reflectivity index across different lithologies
    based on seismic velocities and rock density
    1) Train a Neural Network based on labeled well-logs (Vp,Vs, rho, class label = reflectivity)
    2) Cross-validation testing
    3) Compare performance to base model (random and majority class label)
    4) Blind test of trained NN

    input: X_train.npy - np.vstack([vp1, vs1, rho1, vp2, vs2, rho2, theta]).T
        with:
            # Make 'Upper' layers.
            vp1 = vp[:-1]
            vs1 = vs[:-1]
            rho1 = rho[:-1]

            # Make 'Lower' layers.
            vp2 = vp[1:]
            vs2 = vs[1:]
            rho2 = rho[1:]
"""
import matplotlib.pyplot as plt
import numpy as np
#-----scikit--------------------------------
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit# for cross-validatio
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
# classifiers/estimators
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import linear_model
# Performance evaluation
import sklearn.metrics
#=====================0=====================
#            params and files
#===========================================
np.random.seed( 12345)
standard  = True
l_head    = ['Vp', 'Vs', 'Rho']
file_in   = 'data/well_train.npy'
file_test = 'data/well_test.npy'

#estimator = MLPRegressor() #SVR()
#estimator = linear_model.LinearRegression()
estimator  = linear_model.SGDRegressor()
#=====================1=====================
#            load data
#===========================================
m_data  = np.load(file_in)
m_test  = np.load(file_test)

X_train   = m_data[:,0:-1]
a_y_train = m_data[:,-1]

X_test    = m_test[:,0:-1]
a_y_test  = m_test[:,-1]
print( ' Data (nSample, nAttrib): ', X_train.shape, a_y_train.shape)

if standard == True:
    # standardize the target variables here so prediction can be made
    # and transformed back to original range
    mean, sigma = a_y_train.mean(), a_y_train.std()
    a_y_train   = (a_y_train-mean)/sigma
    a_y_test    = (a_y_test-mean)/sigma
#=====================2=====================
#              train ML algorithm
#===========================================
estimator.fit( X_train, a_y_train)
a_y_train_hat = estimator.predict( X_train)
a_y_test_hat  = estimator.predict( X_test)
#=====================3=====================
#               performance eval
#===========================================
R2_test  = round( estimator.score( X_test, a_y_test),2)
print( 'R2: test', R2_test)
if standard == True:
    # transform predictions back to original range
    a_y_train     = a_y_train*sigma + mean
    a_y_train_hat = a_y_train_hat*sigma + mean
    a_y_test      = a_y_test*sigma + mean
    a_y_test_hat  = a_y_test_hat*sigma + mean

R2_train = round(sklearn.metrics.r2_score( a_y_train, a_y_train_hat),2)
print( 'R2: train', R2_train)

print( 'R2: test', sklearn.metrics.r2_score( a_y_test, a_y_test_hat))
a_y_base = a_y_train[np.random.randint(0,len(a_y_train)-1, len(a_y_test))]
print( 'R2: base-model', sklearn.metrics.r2_score( a_y_test, a_y_base))

MSE_train = round(sklearn.metrics.mean_squared_error(  a_y_train, a_y_train_hat),4)
print( "MSE: train", MSE_train)
MSE_test  = round(sklearn.metrics.mean_squared_error(  a_y_test, a_y_test_hat),4)
print( "MSE: test", MSE_test)

# cross-validation to get more accurate learning curve results
print('max train data size: ', len(a_y_train) *.9)# validation is Ntot - maxNtrain
cv = ShuffleSplit( n_splits=10, test_size=0.1, random_state=12345)
a_nTrain, m_train_scores, m_valid_scores =learning_curve(estimator=estimator,
                                                       X=X_train,y=a_y_train,
                                                       scoring= 'r2', #'mean_squared_error', #'r2',
													   train_sizes=np.linspace(0.1, 1.0, 10),
													   cv=cv,n_jobs=-1)
print( 'training data size', a_nTrain)
#=====================4=====================
#                figures
#===========================================
# note that here we plot results as a function of sample not depth
# for simplicity. We could include depth in the training and testing data
# but it should only be used for plotting purposes
plt.figure(1)
ax = plt.subplot( 131)
ax.set_title(f"Train, R2={R2_train}")
ax.plot( a_y_train,     np.arange( len(a_y_train)),  'k-', lw = 3, alpha = .3, label = 'Obs')
ax.plot( a_y_train_hat, np.arange( len(a_y_train)),  'r-', lw = 1, alpha = 1, label = 'Model')

ax.invert_yaxis()
ax.legend()
ax.set_xlabel( 'Reflectivity')
ax.set_ylabel( 'Depth (samples)')

ax = plt.subplot( 132)
ax.set_title(f"Test, R2={R2_test}")
ax.plot( a_y_test, np.arange( len(a_y_test)),  'k-', lw = 3, alpha = .3, label = 'Obs')
ax.plot( a_y_test_hat, np.arange( len(a_y_test)),  'r-', lw = 1, alpha = 1, label = 'Model')
ax.plot( a_y_base, np.arange( len(a_y_base)),  'g-', lw = 1, alpha = 1, label = 'Base')
ax.invert_yaxis()
ax.legend()
ax.set_xlabel( 'Reflectivity')
ax.set_ylabel( '')

ax2= plt.subplot( 133)
ax2.hist( (a_y_test-a_y_test_hat), color = 'b', label = 'misfit')
ax2.set_xlabel( 'misfit')
plt.savefig( f"well_log_MSE.png")

plt.figure(2) #learning curve
ax = plt.subplot(111)
ax.plot( a_nTrain, m_train_scores.mean(axis=1), 'g-', lw = 2, label = 'Training')
ax.plot( a_nTrain, m_valid_scores.mean(axis=1), 'r-', lw = 2, label = 'Validation')
ax.legend( loc = 'upper right')
ax.set_xlabel( 'Size of Training Data')
ax.set_ylabel( 'Score (R2)')
plt.savefig( 'well_log_learning_curve.png')
plt.show()
#=====================5=====================
#         compare to physical model
#===========================================
# import zoeppritz
#
# theta = np.arange(60)
# sample = 76
#
# amp, amp_hat = get_amps( X_blind_unscaled, sample=sample, theta=theta)
#
# fig, ax = plt.subplots(figsize=(10, 4))
#
# ax.plot(theta, amp, label="Ground truth", lw=2)
# ax.plot(theta, amp_hat, label="Network output", lw=2)
#
# ax.axvspan(min_theta, max_theta, facecolor='k', alpha=0.1, lw=0)
# ax.axhline(0, lw=0.75, color='k')
# ax.text(np.mean([min_theta, max_theta]), plt.gca().get_ylim()[1] - 0.005,
#         s="ϑ TRAINING DOMAIN",
#         fontsize=14, color=(0.4, 0.4, 0.4),
#         va='top', ha='center')
#
# ax.grid(color='k', alpha=0.15)
# ax.set_xlabel('Incidence angle, theta [deg]', size=14)
# ax.set_ylabel('Amplitude', size=14)
# ax.tick_params(axis='both', labelsize=12)
# ax.legend(fontsize=14)
#
# plt.tight_layout()
# ms.savefig(fig, 'thetas')
# plt.show()