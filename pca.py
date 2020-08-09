import numpy as np 
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as ss
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoLocator, LinearLocator, MaxNLocator, AutoMinorLocator, ScalarFormatter)
from decimal import Decimal
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler


#####  Load data here ###########
X_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4)) # PCA will be performed on these flux values
data_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4,16))
print (np.shape(X_south))
print (np.shape(data_south))

X_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4)) # PCA will be performed on these flux values
data_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4,10))
print (np.shape(X_north))
print (np.shape(data_north))
X = np.vstack((X_south, X_north))
data = np.vstack((data_south, data_north))
shape_X = np.shape(X)

Y_south = np.loadtxt("ngc_2023_SNR_south.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4))
Y_north = np.loadtxt("ngc_2023_SNR_north.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4))
Y = np.vstack((Y_south, Y_north))

shape_Y = np.shape(Y)
shape_data = np.shape(data)

Total_PAH = X_south[:,0]+X_south[:,1]+X_south[:,2]+X_south[:,3]+X_south[:,4]

find_ind = np.where(Total_PAH == np.amax(Total_PAH))
print (find_ind)



find_value_62 = np.amax(X_south[:,0][13])
find_value_112 = np.amax(X_south[:,1][13])
find_value_77 = np.amax(X_south[:,2][13])
find_value_110 = np.amax(X_south[:,3][13])
find_value_86 = np.amax(X_south[:,4][13])


####### 3 sigma cuts ##########
''' Following block of code replaces the values below 3 sigma level to zero '''
for i in range (0, shape_Y[1]):
    new_array = np.array([])
    a = Y[:,i]
    b = np.where(a < 3)
    e = len(b[0])
    for j in range (0, shape_X[1]):
        c = X[:,j]
        d = np.delete(c, b[0])
        for k in range(0, e):
            f = np.insert(d, b[0][k], 0)
            d = f
        new_array = np.append(new_array, d)
    n = len(new_array)/shape_X[1]
    X_new = np.reshape(new_array, (int(shape_X[1]), int(n))).T
    X = X_new

X_sig_corr = X
print(np.shape(X_sig_corr))
for i in range (0, shape_Y[1]):
    new_array = np.array([])
    a = Y[:,i]
    b = np.where(a < 3)
    e = len(b[0])
    for j in range (0, shape_data[1]):
        c = data[:,j]
        d = np.delete(c, b[0])
        for k in range(0, e):
            f = np.insert(d, b[0][k], 0)
            d = f
        new_array = np.append(new_array, d)
    n = len(new_array)/shape_data[1]
    data_new = np.reshape(new_array, (int(shape_data[1]), int(n))).T
    data = data_new

data_sig_corr = data
print(np.shape(data_sig_corr))

split_FOVs_array = np.vsplit(X_sig_corr, [1972])
data_sig_corr_south = split_FOVs_array[0]
data_sig_corr_north = split_FOVs_array[1]

''' Following block of code removes the zero values and hence the values below 3 sigma level from the data'''
for i in range(0, shape_X[1]):
    new_array = np.array([])
    a = X_sig_corr[:,i]
    b = np.where(a == 0)
    e = len(b[0])
    for j in range (0, shape_X[1]):
        c = X_sig_corr[:,j]
        d = np.delete(c, b[0])
        new_array = np.append(new_array, d)
    n = len(new_array)/shape_X[1]
    X_new = np.reshape(new_array, (int(shape_X[1]), int(n))).T
    X = X_new

X_zer_rem = X 


for i in range(0, shape_data[1]):
    new_array = np.array([])
    a = data_sig_corr[:,i]
    b = np.where(a == 0)
    e = len(b[0])
    for j in range (0, shape_data[1]):
        c = data_sig_corr[:,j]
        d = np.delete(c, b[0])
        new_array = np.append(new_array, d)
    n = len(new_array)/shape_data[1]
    data_new = np.reshape(new_array, (int(shape_data[1]), int(n))).T
    data = data_new

data_zer_rem = data 
print(np.shape(data_zer_rem))


##### Basic statistics of the data ##############
new_array = np.array([])
for i in range(0, shape_X[1]):
	a = X_zer_rem[:,i]
	b = np.mean(a)
	new_array = np.append(new_array, b)
	c = np.std(a)
	new_array = np.append(new_array, c)

statistics = np.reshape(new_array, (shape_X[1], 2)) # First Column is mean and Second column is standard deviation


# Principal Component Analysis
scaler = StandardScaler()
X_zer_rem_std = StandardScaler().fit_transform(X_zer_rem)

pca = PCA(n_components=5, svd_solver='full')
results = pca.fit(X_zer_rem_std)

var_exp = pca.explained_variance_ratio_  # variance explained by Pcs
num_pc = len(var_exp)
for i in range(0, num_pc):
    k = i+1
    print ("Variance explained by PC%s is %s" % (k, var_exp[i]))

coeff = pca.components_
print ('Coefficient of PCs in terms of original variables (row wise)', coeff)

'''this will return a 2d array of the data projected into PCA space'''
transformed_data = pca.fit_transform(X_zer_rem_std)
np.savetxt("PCs_2023.csv", (transformed_data), delimiter=",")


######## Biplots ###############
def draw_vectors(components, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling
    """

    num_columns = len(columns)

    xvector = components[0]
    yvector = components[1]
    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='#4682b4', width=0.0005, head_width=0.02)
        plt.text(xvector[i]*1.07, yvector[i]*1.07, list(columns)[i], color='#4682b4', fontsize = 18)

    return ax

names = [r'$z_{6.2}$', r'$z_{11.2}$', r'$z_{7.7}$', r'$z_{11.0}$', r'$z_{8.6}$']


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax = draw_vectors(coeff, names)


plt.xlabel(r'$PC_{1}$', fontsize = 18)
plt.ylabel(r'$PC_{2}$', fontsize = 18)
ax.tick_params(direction='in', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize =18)
ax.tick_params(direction='in', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)


majorLocator = MultipleLocator(0.25) 
ax.xaxis.set_major_locator(majorLocator)
minorLocator = MultipleLocator(0.05)
ax.xaxis.set_minor_locator(minorLocator)
majorLocator = MultipleLocator(0.25) 
ax.yaxis.set_major_locator(majorLocator)
minorLocator = MultipleLocator(0.05)
ax.yaxis.set_minor_locator(minorLocator)
plt.ylim((-1,1))
plt.xlim((-1,1))
#plt.savefig('PC1_PC2_biplot.eps', bbox_inches='tight')
plt.show()

def characteristic_spectrum(): 
    n0 = input('Principal Component for which Characteristic spectrum is to be generated:')
    n = int(n0)-1
    if n > 4 or n<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
        
    c = coeff[n,:].T
    a_0 = (c[0]*statistics[0,1])+ statistics[0,0]
    a_1 = (c[1]*statistics[1,1])+ statistics[1,0]
    a_2 = (c[2]*statistics[2,1])+ statistics[2,0]
    a_3 = (c[3]*statistics[3,1])+ statistics[3,0]
    a_4 = (c[4]*statistics[4,1])+ statistics[4,0]
    
    a_0 = a_0
    a_1 = a_1
    a_2 = a_2
    a_3 = a_3
    a_4 = a_4
    FWHM = np.array([0.19, 0.45, 0.29, 0.15, 0.24])
    sigma = FWHM/2.355
    print (sigma)
    mu = [6.2, 7.7, 8.6, 11.0, 11.2]
    peak_intensity = [a_0, a_2, a_4, a_3, a_1]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range (0, 5):
        x = np.linspace(mu[i] - 5*sigma[i], mu[i] + 5*sigma[i], 100)
        y_pdf = (ss.norm.pdf(x, mu[i], sigma[i]))*peak_intensity[i]
        plt.plot(x, y_pdf, color='#4682b4', linewidth=2)
        
    ax = plt.axes()  
    plt.xlabel(r'Wavelength ($\mu$m)', fontsize = 18)
    plt.ylabel(r'Intensity [$X 10^{-5}$ $W m^{-2} sr^{-1}$]', fontsize = 18)
    plt.axhline(y=0.0, linestyle ='solid', linewidth=1, color = 'k')
    ax.tick_params(direction='in', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize =18)
    ax.tick_params(direction='in', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)
    
    majorLocator = MultipleLocator(1) 
    ax.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.2)
    ax.xaxis.set_minor_locator(minorLocator)
    majorLocator = AutoLocator()
    ax.yaxis.set_major_locator(majorLocator)
    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    offset = ax.get_yaxis().get_offset_text()
    offset.set_visible(False)
    plt.ylim(0, 0.000035)
    #plt.savefig('characteristic_PAH_spectrum_PC' + str(n+1) + 'variable_FWHM_ngc2023' + '.eps', bbox_inches='tight')
    plt.show()

characteristic_spectrum()


def eig_spectrum(): 
    n0 = input('Principal Component for which eigen spectra is to be generated:')
    n = int(n0)-1
    
    if n > 4 or n<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
        
    c =  coeff[n,:] 
    sigma = [0.19, 0.45, 0.29, 0.15, 0.24]
    mu = [6.2, 7.7, 8.6, 11.0, 11.2]
    peak_intensity = [c[0], c[2], c[4], c[3], c[1]]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range (0, 5):
        x = np.linspace(mu[i] - 5*sigma[i], mu[i] + 5*sigma[i], 100)
        y_pdf = (ss.norm.pdf(x, mu[i], sigma[i]))*peak_intensity[i]
        plt.plot(x, y_pdf, color='#4682b4', linewidth=2)
        
    plt.xlabel(r'Wavelength ($\mu$m)', fontsize = 18)
    plt.ylabel('Intensity (arbitrary units)', fontsize = 18)
    plt.axhline(y=0.0, linestyle ='solid', linewidth=1, color = 'k')
    ax.tick_params(direction='in', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize =18)
    ax.tick_params(direction='in', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)
    majorLocator = MultipleLocator(1) 
    ax.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.2)
    ax.xaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(1.0) 
    ax.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.2)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.xlim(5,12.5)
    plt.ylim(0, 1.5)
    #plt.savefig('eigen_spectrum_PC' + str(n+1) + '_NGC2023' + '.eps', bbox_inches='tight')	
    plt.show()


eig_spectrum()


slope_reg1 = ((4.56928*10**(-6))-(1.78478*10**(-6)))/(3.97106-0.0239173)
intercept_reg1 = (1.78478*10**(-6))-(slope_reg1*(0.0239173))
regression_line_1 = [(slope_reg1*x)+intercept_reg1 for x in PC]


slope_reg2 = ((7.58939*10**(-6))-(7.12428*10**(-8)))/(3.86474+4.27541)
intercept_reg2 = (7.58939*10**(-6))-(slope_reg2*(3.86474))
regression_line_2 = [(slope*x)+intercept for x in PC]

######## Correlation Plots #######

PC = transformed_data[:,0]   ### X-axis variable...transformed_data[:,0] = PC1; transformed_data[:,0] = PC2 etc..

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(34, 34))  #### Fig size for two columns: (30, 14)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

list_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
mapping = {'0' : X_zer_rem[:,0], 
           '1': X_zer_rem[:,2],
           '2': X_zer_rem[:,4],
           '3': X_zer_rem[:,3],
           '4': X_zer_rem[:,1],
           '5': (X_zer_rem[:,0]+X_zer_rem[:,1]+X_zer_rem[:,2]+X_zer_rem[:,3]+X_zer_rem[:,4]),
           '6': (X_zer_rem[:,0]/X_zer_rem[:,1]),
           '7': (X_zer_rem[:,2]/X_zer_rem[:,1]),
           '8': (X_zer_rem[:,4]/X_zer_rem[:,1]),
           '9': (X_zer_rem[:,3]/X_zer_rem[:,1]),
           '10': (X_zer_rem[:,0]/X_zer_rem[:,4]),
           '11': (X_zer_rem[:,2]/X_zer_rem[:,4]),
           '12': (X_zer_rem[:,0]/X_zer_rem[:,3]),
           '13': (X_zer_rem[:,2]/X_zer_rem[:,3]),
           '14': (X_zer_rem[:,4]/X_zer_rem[:,3]),
           '15': (X_zer_rem[:,0]/X_zer_rem[:,2])}
mapping_ylabels = {'0' : r'6.2 $\mu$m flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '1': r'7.7 $\mu$m flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '2': r'8.6 $\mu$m flux [$X 10^{-6}$ $W m^{-2} sr^{-1}$]',
                   '3': r'11.0 $\mu$m flux [$X 10^{-7}$ $W m^{-2} sr^{-1}$]', 
                   '4': r'11.2 $\mu$m flux [$X 10^{-6}$ $W m^{-2} sr^{-1}$]', 
                   '5': r'Total PAH flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '6': r' 6.2/11.2',
                   '7': r' 7.7/11.2',
                   '8': r' 8.6/11.2',
                   '9': r' 11.0/11.2 [$X 10^{-1}$]',
                   '10': r' 6.2/8.6',
                   '11': r' 7.7/8.6 [$X 10^{1}$]',
                   '12': r' 6.2/11.0 [$X 10^{1}$]',
                   '13': r' 7.7/11.0 [$X 10^{2}$]',
                   '14': r' 8.6/11.0 [$X 10^{1}$]',
                   '15': r' 6.2/7.7 [$X 10^{-1}$]'}

for i in range(0, 4):
    for j in range(0, 4):
        if i == 1:
            n = j+4
        elif i == 2:
            n = j+8
        elif i == 3:
            n = j+12
        else:
            n = j
            
        X_data = mapping[list_symbols[n]]
        
        ax[i,j].plot(PC, X_data, color='royalblue', marker = 'o', markersize=4, linestyle='None')
        
        if n == 4:
            slope_reg1 = ((1.99758*10**(-7))-(6.91469*10**(-6)))/(-4.29535-2.64206)
            intercept_reg1 = (1.99758*10**(-7))-(slope_reg1*(-4.29535))
            regression_line_1 = [(slope_reg1*x)+intercept_reg1 for x in PC]
            slope_reg2 = ((7.56658*10**(-7))-(7.18243*10**(-6)))/(-3.30524-4.3166)
            intercept_reg2 = (7.18243*10**(-6))-(slope_reg2*(4.3166))
            regression_line_2 = [(slope_reg2*x)+intercept_reg2 for x in PC]
            ax[i,j].plot(PC, regression_line_1, color = 'black')
            ax[i,j].plot(PC, regression_line_2, color = 'black')
            
        r_value, p_value = ss.pearsonr(PC, X_data)
        corr_coeff = round((r_value**(2)), 4)
        if n < 6:
            ax[i,j].text(0.39*max(PC), 1.05*min(X_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
        elif n > 5 and n < 10:
            ax[i,j].text(0.39*max(PC), 0.90*max(X_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
        else:
            ax[i,j].text(0.39*max(PC), 1.05*min(X_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
      
        ax[i,j].set_ylabel(mapping_ylabels[list_symbols[n]],fontsize=27)
        ax[i,j].set_xlabel(r'$PC_{2}$', fontsize=27)  
        ax[i,j].tick_params(direction='in', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize = 27)
        ax[i,j].tick_params(direction='in', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)
        majorLocator = MultipleLocator(1)
        minorLocator = AutoMinorLocator()
        ax[i,j].xaxis.set_major_locator(majorLocator)
        ax[i,j].xaxis.set_minor_locator(minorLocator)
        majorLocator = AutoLocator()
        minorLocator = AutoMinorLocator()
        ax[i,j].yaxis.set_major_locator(majorLocator)
        ax[i,j].yaxis.set_minor_locator(minorLocator)
        ax[i,j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        offset = ax[i,j].get_yaxis().get_offset_text()
        offset.set_visible(False)

#plt.savefig('PC2_correlation_ngc_2023_pearson_pvalue.eps', bbox_inches='tight'  )
plt.show()


######## Calculate confidence intervals######################
PC = transformed_data[:,1]
X_data = X_zer_rem[:,0]/X_zer_rem[:,1]
r_value, p_value = ss.pearsonr(PC, X_data)

alpha = 0.05 / 2 # Two-tail test
z_critical = ss.norm.ppf(1 - alpha)
print (z_critical)

r = r_value # Pearson's r from sampled data
print (r)
z_prime = 0.5 * np.log((1 + r) / (1 - r))

n = len(PC) # Sample size
se = 1 / np.sqrt(n - 3) # Sample standard error

CI_lower = z_prime - z_critical * se
CI_upper = z_prime + z_critical * se

print (CI_lower)
print (CI_upper)


################################################
PC = transformed_data[:,0] 
cm = plt.cm.get_cmap('viridis')  
z =  X_zer_rem[:,3]/X_zer_rem[:,1]    # Color bar for PC1(PC2)
#z = -transformed_data[:,1]   # Color bar for PC1(PC2)

fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (34, 38))
#fig, ax = plt.subplots(nrows=4, ncols=4)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

#fig.subplots_adjust(wspace=0.3)
#fig.subplots_adjust(hspace=1.0)
list_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
mapping = {'0' : X_zer_rem[:,0], 
           '1': X_zer_rem[:,2],
           '2': X_zer_rem[:,4],
           '3': X_zer_rem[:,3],
           '4': X_zer_rem[:,1],
           '5': (X_zer_rem[:,0]+X_zer_rem[:,1]+X_zer_rem[:,2]+X_zer_rem[:,3]+X_zer_rem[:,4]),
           '6': (X_zer_rem[:,0]/X_zer_rem[:,1]), 
           '7': (X_zer_rem[:,2]/X_zer_rem[:,1]),
           '8': (X_zer_rem[:,4]/X_zer_rem[:,1]),
           '9': (X_zer_rem[:,3]/X_zer_rem[:,1]),
           '10': (X_zer_rem[:,0]/X_zer_rem[:,4]),
           '11': (X_zer_rem[:,2]/X_zer_rem[:,4]), 
           '12': (X_zer_rem[:,0]/X_zer_rem[:,3]), 
           '13': (X_zer_rem[:,2]/X_zer_rem[:,3]),
           '14': (X_zer_rem[:,4]/X_zer_rem[:,3]),
           '15': (X_zer_rem[:,0]/X_zer_rem[:,2]) }
mapping_ylabels = {'0' : r'6.2 $\mu$m flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '1': r'7.7 $\mu$m flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '2': r'8.6 $\mu$m flux [$X 10^{-6}$ $W m^{-2} sr^{-1}$]',
                   '3': r'11.0 $\mu$m flux [$X 10^{-7}$ $W m^{-2} sr^{-1}$]', 
                   '4': r'11.2 $\mu$m flux [$X 10^{-6}$ $W m^{-2} sr^{-1}$]', 
                   '5': r'Total PAH flux [$X 10^{-5}$ $W m^{-2} sr^{-1}$]',
                   '6': r' 6.2/11.2',
                   '7': r' 7.7/11.2',
                   '8': r' 8.6/11.2',
                   '9': r' 11.0/11.2 [$X 10^{-1}$]',
                   '10': r' 6.2/8.6',
                   '11': r' 7.7/8.6 [$X 10^{1}$]',
                   '12': r' 6.2/11.0 [$X 10^{1}$]',
                   '13': r' 7.7/11.0 [$X 10^{2}$]',
                   '14': r' 8.6/11.0 [$X 10^{1}$]',
                   '15': r' 6.2/7.7 [$X 10^{-1}$]'}

images = []
for i in range(0, 4):
    for j in range(0, 4):
        if i == 1:
            n = j+4
        elif i == 2:
            n = j+8
        elif i == 3:
            n = j+12
        else:
            n = j
        X_data = mapping[list_symbols[n]]
        images.append(ax[i,j].scatter(PC, X_data, c=z, cmap= cm))
        if n == 4:
            rect = Rectangle((-4.6, 0), width = 0.2, height = 0.000002, angle=(10))
            ax[i,j].add_patch(rect)
            #ax[i,j].patches.Rectangle(3.90461, 4.65496*10**(-6), 2, 2, 0)

        
        #sc = ax[i,j].scatter(PC, X_data, c=z, cmap= cm)
        #plt.colorbar(sc)
        #ax[i,j].plot(PC, X_data, c=z, cmap= cm )
        #ax[i,j].scatter(PC, X_data, c=z, cmap= cm)
        ax[i,j].set_ylabel(mapping_ylabels[list_symbols[n]], fontsize = 27)
        ax[i,j].set_xlabel(r'$PC_{1}$', fontsize = 27)  
        ax[i,j].tick_params(direction='in', which = 'major', bottom= True, top= True, left= True, right= True, length=12, labelsize = 27) 
        ax[i,j].tick_params(direction='in', which = 'minor', bottom= True, top= True, left= True, right= True, length=6)
        majorLocator = MultipleLocator(1)
        minorLocator = AutoMinorLocator()
        ax[i,j].xaxis.set_major_locator(majorLocator)
        ax[i,j].xaxis.set_minor_locator(minorLocator)
        majorLocator = AutoLocator()
        minorLocator = AutoMinorLocator()
        ax[i,j].yaxis.set_major_locator(majorLocator)
        ax[i,j].yaxis.set_minor_locator(minorLocator)
        ax[i,j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        offset = ax[i,j].get_yaxis().get_offset_text()
        offset.set_visible(False)
        if n == 0:
            ax[i,j].set_ylim(0, 1.2*10**(-5))
        elif n == 1:
            ax[i,j].set_ylim(0, 2.25*10**(-5))
        elif n == 2:
            ax[i,j].set_ylim(0, 4.5*10**(-6))
        elif n == 3:
            ax[i,j].set_ylim(0, 6.0*10**(-7))
        elif n == 4:
            ax[i,j].set_ylim(0, 8.0*10**(-6))
        elif n == 5:
            ax[i,j].set_ylim(0, 4.5*10**(-5))
        else:
            pass

#ax=ax.ravel().tolist()
cb = plt.colorbar(images[0], ax=ax.ravel().tolist(), orientation = 'horizontal', pad = 0.035)
cb.ax.tick_params(labelsize=27)
cb.ax.set_xlabel('11.0/11.2', fontsize = 27)


#plt.savefig('Pop_division_on_PC1_110_112_ngc_2023.eps', bbox_inches='tight' )
plt.show()

####################### Spatial Map ###########################
## First Let's insert zeros back in the data
X = X_sig_corr

for i in range(0, shape_X[0]):
    new_array = np.array([])
    a = X[i,:]
    b = np.where(a == 0)
    c = len(b[0])
    if c >=1:
        for j in range(0, shape_X[1]):
            d = transformed_data[:,j]
            e = np.insert(d,i,0)
            new_array = np.append(new_array, e)
        n = len(new_array)/shape_X[1]
        transformed_data = np.reshape(new_array, (int(shape_X[1]), int(n))).T
PC_new = transformed_data

split_FOVs_array = np.vsplit(PC_new, [1972])
PC_south = split_FOVs_array[0]
PC_north = split_FOVs_array[1]

########## Spatial Map ##################################

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Contour plots of PCs
k1_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (1)) #11.2
k2_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (2)) #7.7
r1_south = np.reshape(k1_south, (34, 58))
r2_south = np.reshape(k2_south, (34, 58))


m_south = np.reshape(-PC_south[:,0], (34, 58))
mask_array_south = m_south.astype(np.float)
#plot_south = r1_south
plot_south = np.reshape(-PC_south[:,0], (34, 58))
unmask_array_south = plot_south.astype(np.float)
unmask_array_south[np.where(mask_array_south == 0)] = np.nan
mask_array_south[np.where(mask_array_south != 0)] = np.nan

z_south = unmask_array_south.T

z1_south = r1_south.T
z2_south = r2_south.T
z3_south = mask_array_south.T

xlist = np.linspace(1,34,58)
ylist = np.linspace(1,34,58)
x,y = np.meshgrid(xlist, ylist)

#k1_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (1))
#k2_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (2))

#r1_north = np.reshape(k1_north, (20, 30))
#r2_north = np.reshape(k2_north, (20, 30))

#m_north = np.reshape(-PC_north[:,0], (20, 30))
#mask_array_north = m_north.astype(np.float)
#plot_north = np.reshape(-PC_north[:,0], (20, 30))
#unmask_array_north = plot_north.astype(np.float)
#unmask_array_north[np.where(mask_array_north == 0)] = np.nan
#mask_array_north[np.where(mask_array_north != 0)] = np.nan

#z_north = unmask_array_north.T

#z1_north = r1_north.T
#z2_north = r2_north.T
#z3_north = mask_array_north.T


fig, ax = plt.subplots(1, 1)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')

fig.subplots_adjust(wspace=0.25)

ax.imshow(z3_south, cmap='gray', vmin=0,vmax=1)
im = ax.imshow(z_south, cmap='viridis')

cb = fig.colorbar(im, ax=ax, format= ScalarFormatter(useOffset=None, useMathText=True))
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
cb.ax.tick_params(labelsize=32)
angle_x = math.radians(266.4)
angle_y = math.radians(90+(266.4))


ax.contour(z1_south, levels = [0.00000366, 0.00000464, 0.00000564, 0.00000678], colors=('k',),linestyles=('solid'), linewidths = 3)
ax.contour(z2_south, levels = [0.0000140, 0.0000156, 0.0000170, 0.0000190], colors=('white',),linestyles=('solid'), linewidths = 3)
ax.annotate(r"$S^{'}$", xy=(2.3, 1.5), xytext=(3.9, 7.2), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
ax.annotate('SE', xy=(1.8, 13.7), xytext=(9.8, 7.9), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
ax.annotate('SSE', xy=(2, 28), xytext=(1.8, 42), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
ax.annotate('S', xy=(16, 31), xytext=(25, 21), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
ax.arrow(30, 50, 5*math.cos(angle_x), 5*math.sin(angle_x), color='white', width=0.0005, head_width=1)
ax.arrow(30, 50, -5*math.cos(angle_y), -5*math.sin(angle_y), color='white', width=0.0005, head_width=1)
ax.text(28, 41, 'N', color='white', fontsize = 20)
ax.text(20, 52, 'E', color='white', fontsize = 20)
ax.set_xlabel('Pixel Number', fontsize= 32)
ax.set_ylabel('Pixel Number', fontsize= 32)
ax.tick_params(direction='out', which = 'major', bottom= True, left= True, top=True, right=True, length = 18, labelsize =32)
ax.tick_params(direction='out', which = 'minor', bottom= True, left= True, top=True, right=True, length = 10)
majorLocator = MultipleLocator(10)
ax.xaxis.set_major_locator(majorLocator)
ax.yaxis.set_major_locator(majorLocator)
minorLocator = MultipleLocator(1.0)
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
#ax.imshow(z3_north, cmap='gray', vmin=0,vmax=1)
#im = ax.imshow(z_north, cmap='viridis')

#cb = fig.colorbar(im, ax=ax)

#cb.ax.tick_params(labelsize=32)
#ax.contour(z1_north, levels = [0.00000142, 0.00000157,0.00000175, 0.00000199], colors=('k',),linestyles=('solid'), linewidths = 3)
#ax.contour(z2_north, levels = [0.00000554, 0.00000630,0.00000680, 0.00000720], colors=('white',),linestyles=('solid'), linewidths = 3)
#ax.annotate('N', xy=(8, 7), xytext=(7, 4), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
#ax.annotate('NW', xy=(12, 17), xytext=(7, 20), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=32)
#ax.set_xlabel('Pixel Number', fontsize=32)
#ax.set_ylabel('Pixel Number', fontsize=32)
#ax.tick_params(direction='out', which = 'major', bottom= True, left= True, top=True, right=True, length =18, labelsize = 32)
#ax.tick_params(direction='out', which = 'minor', bottom= True, left= True, top=True, right=True, length = 10)
#majorLocator = MultipleLocator(5)
#ax.xaxis.set_major_locator(majorLocator)
#ax.yaxis.set_major_locator(majorLocator)
#minorLocator = MultipleLocator(0.5)
#ax.xaxis.set_minor_locator(minorLocator)
#ax.yaxis.set_minor_locator(minorLocator)



#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()
#fig.savefig('PC1_spatial_map_south.png', bbox_inches='tight')
plt.show()

#fig.savefig('PC1_spatial_map_south.eps', bbox_inches='tight')
