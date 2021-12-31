#opt/anaconda3/bin/python3.7
"""
    compute reflectivity index across different lithologies
    based on seismic velocities and rock density
        First step in machine learning workflow
        - data screening and pre-processing
        - standardizing, train/test/valid data split
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#=====================0=====================
#            files and params
#===========================================
well_file     = 'data/well_log.npy'
test_size     = 0.2#10 to 30 percent
addBott_layer = True
standard      = True
#=====================1=====================
#                 load data
#===========================================
m_data    = np.load( well_file)
# create depth vector
a_z       = m_data[:,0]
#a_z = np.linspace( zmin, zmax, len(m_data[0]))
X         = m_data[:,1:-1]
a_y       = m_data[:,-1]
print( ' Data (nSample, nAttrib): ', a_z.shape, X.shape, a_y.shape)
# add context - add subsequent layer Vp, Vs, and rho as new attributes
if addBott_layer == True:
    X   = np.vstack(( X[0:-1,:].T, X[1::,:].T)).T
    a_y = a_y[0:-1]
    a_z = a_z[0:-1]
#=====================2=====================
#        train/test split, standardize
#===========================================
X_train, X_test, y_train, y_test = train_test_split(X, a_y,
                                                    test_size=test_size,
                                                    random_state=12345)
print('train size: ', X_train.shape, 'test size', X_test.shape)
# standardize training and testing using same mean and sigma
if standard == True:
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train= scaler.transform( X_train)
    X_test = scaler.transform( X_test)
#=====================3=====================
#         save data files as npy binary
#===========================================
print( np.vstack(( X_train.T, y_train)).T.shape)
np.save( 'data/well_train.npy', np.vstack(( X_train.T, y_train)).T)
np.save( 'data/well_test.npy',  np.vstack(( X_test.T,  y_test)).T)
#=====================4=====================
#           figures:    raw data
#===========================================
#plot training data and class labels
plt.figure(1)
#vp1, vs1, rho1, vp2, vs2, rho2, theta
ax = plt.subplot( 141)
ax.plot( X[:,0]*1e-3, a_z, c = 'C0', label = 'Vp' )
ax.invert_yaxis()
ax.legend( loc = 'upper center', frameon=True)
ax.set_xlabel( 'Vp (km/s)')
ax.set_ylabel( 'Depth (m)')

ax2 = plt.subplot( 142)
ax2.plot( X[:,1]*1e-3, a_z, c = 'C1', label = 'Vs' )
ax2.invert_yaxis()
ax2.legend( loc = 'upper center', frameon=True)
ax2.set_xlabel( 'Vs (km/s)')

ax3 = plt.subplot( 143)
ax3.plot( X[:,2], a_z, c = 'C2', label = 'rho' )
ax3.invert_yaxis()
ax3.legend( loc = 'upper center', frameon=True)
ax3.set_xlabel( 'Rho (kg/m$^3$)')

ax4 = plt.subplot( 144)
ax4.plot( a_y, a_z, c = 'C3', label = 'Reflectivity' )
ax4.invert_yaxis()
ax4.legend( loc = 'upper center', frameon=True)
ax4.set_xlabel( 'Reflectivity')
#=====================4=====================
#          correlation matrix with pandas
#===========================================
l_head = ['Depth','Vp', 'Vs', 'rho', 'reflec']
df = pd.DataFrame( m_data, columns=l_head)

pd.plotting.scatter_matrix(df, alpha=0.2, color = 'C1')
#TODO: scatter plot with seaborn

# correlation matrix
corrMatrix = df.corr()
print( corrMatrix)
plt.figure(3)
axCM = plt.subplot( 111)
plot3 = axCM.pcolor( corrMatrix, cmap = plt.cm.hot_r)
plt.colorbar( plot3, label = 'CC')
axCM.set_xticks( axCM.get_xticks()[0:-1]+.5)
axCM.set_yticks( axCM.get_yticks()[0:-1]+.5)
axCM.set_xticklabels( l_head)
axCM.set_yticklabels( l_head)
plt.show()