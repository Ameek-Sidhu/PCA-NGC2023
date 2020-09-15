import sys 
import math
import numpy as np
import scipy.stats as ss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator, ScalarFormatter)

#####  Load data here ###########
flux_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4))
flux_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4)) 
snr_south = np.loadtxt("ngc_2023_SNR_south.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4))
snr_north = np.loadtxt("ngc_2023_SNR_north.csv", delimiter = ',', skiprows = 1, usecols = (0,1,2,3,4))

flux = np.vstack((flux_south, flux_north))
snr = np.vstack((snr_south, snr_north))

shape_data = np.shape(flux)
############################################
def sigma_corr_data(flux, snr, shape_data):
    """
    NAME:
    sigma_corr_data
    
    DESCRIPTION:
    This function performs the 3 sigma cuts to the data.
    For the fluxes where SNR<3, this function will remove those values from the data. 
    
    INPUT:
    flux = 2-D array
    snr = 2-D array
    shape_data = shape of the flux array
    
    RETURN:
    ind_sigma_flag_array = 1-D array containing indices which are removed from the data
    flux_sig_corr = 2-D array containing sigma corrected data
   
    
    """
    
    ind_sigma_flag_array = np.ones(int(shape_data[0]))
    for i in range(0, shape_data[1]):
        ind_sigma_flag_array = np.where(snr[:,i]<3, 0, ind_sigma_flag_array)
        
    ind_sigma_corr = np.where(ind_sigma_flag_array==0)


    new_array = np.array([])
    for i in range (0, shape_data[1]):
        a = np.delete(flux[:,i], ind_sigma_corr)
        new_array = np.append(new_array, a)
    
    n = len(new_array)/shape_data[1]
    flux_sig_corr = np.reshape(new_array, (int(shape_data[1]), int(n))).T
        
    return ind_sigma_flag_array, flux_sig_corr
############################################

def stats_data(flux_sig_corr, shape_data):
    """
    NAME:
    stats_data
    
    DESCRIPTION:
    This function calculates mean and standard deviation of PAH fluxes
    
    INPUT:
    flux_sig_corr = 2-D array containing fluxes
    shape_data = array containing shape of the data 
    
    RETURN:
    statistics = 2-D array containing mean and standaed deviation
    
    """
    new_array = np.array([])
    for i in range(0, shape_data[1]):
        new_array = np.append(new_array, np.mean(flux_sig_corr[:,i]))
        new_array = np.append(new_array, np.std(flux_sig_corr[:,i]))

    statistics = np.reshape(new_array, (shape_data[1], 2)) # First Column is mean and Second column is standard deviation
    return statistics
############################################    
def PCA_PAH_flux(flux_sig_corr, shape_data, standardize_data = True):
    """
    NAME:
    PCA_PAH_flux
    
    DESCRIPTION:
    This funtion will perform Principal Component Analysis
    of PAH fluxes.
    
    INPUT:
    flux_sig_corr = 2-D array of PAH fluxes.
    shape_data = shape of the data
    
    INPUT KEYWORD PARAMETERS:
    standardize_data- You may want to standardize the original variables before performing PCA
    so that the standardized variables have mean = 0 and standard deviation = 1.
    By default data is standardized before performing PCA.
    
    RETURN:
    coeff: Coefficient of PCs in terms of original variables.
    transformed_data: 2-D array of the original data projected into PCA space.
    
    """
    if standardize_data == True:
        scaler = StandardScaler()
        flux_sig_corr_std = StandardScaler().fit_transform(flux_sig_corr)
    else:
        flux_sig_corr_std = flux_sig_corr
        
    pca = PCA(n_components=shape_data[1], svd_solver='full')
    results = pca.fit(flux_sig_corr_std)

    var_exp = pca.explained_variance_ratio_  # variance explained by PCs
    num_pc = len(var_exp)
    for i in range(0, num_pc):
        k = i+1
        print ("Variance explained by PC%s is %s" % (k, var_exp[i]))

    coeff = pca.components_
    print ('Coefficient of PCs in terms of original variables (row wise)', coeff)
    
    transformed_data = pca.fit_transform(flux_sig_corr_std)
    
    return coeff, transformed_data
############################################   
list_symbols_PCs = ['0', '1', '2', '3', '4']
mapping_labels_PCs = {'0' : r'$PC_{1}$',
                   '1': r'$PC_{2}$',
                   '2': r'$PC_{3}$',
                   '3': r'$PC_{4}$', 
                   '4': r'$PC_{5}$'}
                   
list_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
mapping = {'0' : flux_sig_corr[:,0], 
           '1': flux_sig_corr[:,2],
           '2': flux_sig_corr[:,4],
           '3': flux_sig_corr[:,3],
           '4': flux_sig_corr[:,1],
           '5': (flux_sig_corr[:,0]+flux_sig_corr[:,1]+flux_sig_corr[:,2]+flux_sig_corr[:,3]+flux_sig_corr[:,4]),
           '6': (flux_sig_corr[:,0]/flux_sig_corr[:,1]),
           '7': (flux_sig_corr[:,2]/flux_sig_corr[:,1]),
           '8': (flux_sig_corr[:,4]/flux_sig_corr[:,1]),
           '9': (flux_sig_corr[:,3]/flux_sig_corr[:,1]),
           '10': (flux_sig_corr[:,0]/flux_sig_corr[:,4]),
           '11': (flux_sig_corr[:,2]/flux_sig_corr[:,4]),
           '12': (flux_sig_corr[:,0]/flux_sig_corr[:,3]),
           '13': (flux_sig_corr[:,2]/flux_sig_corr[:,3]),
           '14': (flux_sig_corr[:,4]/flux_sig_corr[:,3]),
           '15': (flux_sig_corr[:,0]/flux_sig_corr[:,2])}
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

######## Biplots ###############
def biplots(components, names, sv=False):
    """
    NAME:
    BIPLOTS
    
    DESCRIPTION:
    This funtion will project original variables
    onto the reference frame defined by principal components.
    
    INPUT:
    components = coefficients of PCs in terms of original variables.
    names = array of names of original variables.
    
    INPUT KEYWORD PARAMETERS:
    sv- You may want to save the file.
    By default figure is not saved.
    
    """
    
    n_x = input('Principal Component on x-axis:')
    n_x = int(n_x)-1
    if n_x > 4 or n_x<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
    else:
        pass
        
    n_y = input('Principal Component on y-axis:')
    n_y = int(n_y)-1
    if n_y > 4 or n_y<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
    elif n_y==n_x:
        print ('Error! Principal Component on y-axis should be different than x-axis')
        sys.exit()
    else:
        pass
    num_columns = len(names)

    xvector = components[n_x]
    yvector = components[n_y]
    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='#4682b4', width=0.0005, head_width=0.02)
        plt.text(xvector[i]*1.07, yvector[i]*1.07, list(names)[i], color='#4682b4', fontsize = 18)
   
    ax.set_xlabel(mapping_labels_PCs[list_symbols_PCs[n_x]],fontsize=18)
    ax.set_ylabel(mapping_labels_PCs[list_symbols_PCs[n_y]],fontsize=18)
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
    if sv==True:
        plt.savefig('PC'+str(n_x+1)+'_PC'+str(n_y+1)+'_biplot.eps', bbox_inches='tight')
    else:
        pass
    plt.show()

    return 
############################################
names = [r'$z_{6.2}$', r'$z_{11.2}$', r'$z_{7.7}$', r'$z_{11.0}$', r'$z_{8.6}$']
############################################

def eig_spectrum(components, peak_pos, sigma_gauss, sv=False): 
    """
    NAME:
    eig_spectrum
    
    DESCRIPTION:
    Eigen spectrum corresponding to principal component. UPDATE IT.....
    
    INPUT:
    components = 
    peak_pos = 1-D array of peak positions of PAH bands in units of microns.
    sigma_gauss = 1-D array of standard deviations for the gaussians of PAH bands
    
    INPUT KEYWORD PARAMETERS:
    sv- You may want to save the file.
    By default figure is not saved.
    
    """
    if len(peak_pos) != len(sigma_gauss):
        print ('Error! peak_pos and sigma_gauss arrays should have same shape')
        sys.exit()
    else:
        pass
        
    n0 = input('Principal Component for which eigen spectra is to be generated:')
    n_PC = int(n0)-1
    
    if n_PC > 4 or n_PC < 0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
    else:
        pass
        
    c =  components[n_PC,:] 
    mu = peak_pos
    sigma = sigma_gauss
    peak_intensity = [c[0], c[2], c[4], c[3], c[1]]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes()
    
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
    majorLocator = AutoLocator()
    ax.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.2)
    ax.yaxis.set_minor_locator(minorLocator)
    plt.xlim(5,12.5)
    if n_PC==0:
        plt.ylim(0, 2.5)
    elif n_PC==1:
        plt.ylim(-3.5, 3.5)
    else:
        pass
    if sv==True:
        plt.savefig('eigen_spectrum_PC' + str(n+1) + '_NGC2023' + '.eps', bbox_inches='tight')
    else:
        pass
    plt.show()
    
    return
############################################
peak_pos = np.array([6.2, 7.7, 8.6, 11.0, 11.2])
sigma_gauss = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
############################################

def characteristic_spectrum(components, peak_pos, FWHM_gauss, statistics, sv=False): 
    """
    NAME:
    characteristic_spectrum
    
    DESCRIPTION:
    UPDATE IT.....
    
    INPUT:
    components = 
    peak_pos = 1-D array of peak positions of PAH bands in units of microns.
    FWHM_gauss = 1-D array of Full width at half maximum for the gaussians of PAH bands.
    statistics = 
    
    INPUT KEYWORD PARAMETERS:
    sv- You may want to save the file.
    By default figure is not saved.
    
    """
    if len(peak_pos) != len(FWHM_gauss):
        print ('Error! peak_pos and sigma_gauss arrays should have same shape')
        sys.exit()
    else:
        pass
        
    n0 = input('Principal Component for which Characteristic spectrum is to be generated:')
    n_PC = int(n0)-1
    if n_PC > 4 or n_PC<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
    else:
        pass
        
    c = components[n_PC,:].T
    a_0 = (c[0]*statistics[0,1])+ statistics[0,0]
    a_1 = (c[1]*statistics[1,1])+ statistics[1,0]
    a_2 = (c[2]*statistics[2,1])+ statistics[2,0]
    a_3 = (c[3]*statistics[3,1])+ statistics[3,0]
    a_4 = (c[4]*statistics[4,1])+ statistics[4,0]
    

    sigma = FWHM_gauss/2.355
    mu = peak_pos
    peak_intensity = [a_0, a_2, a_4, a_3, a_1]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes() 
    
    for i in range (0, 5):
        x = np.linspace(mu[i] - 5*sigma[i], mu[i] + 5*sigma[i], 100)
        y_pdf = (ss.norm.pdf(x, mu[i], sigma[i]))*peak_intensity[i]
        plt.plot(x, y_pdf, color='#4682b4', linewidth=2)
        
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
    plt.xlim(5,12.5)
    if n_PC==0:
        plt.ylim(0, 0.000035)
    else:
        pass
    if sv==True:
        plt.savefig('characteristic_PAH_spectrum_PC' + str(n+1) + 'variable_FWHM_ngc2023' + '.eps', bbox_inches='tight')
    else:
        pass
    plt.show()
    

    return 
############################################
peak_pos = np.array([6.2, 7.7, 8.6, 11.0, 11.2])
FWHM_gauss = np.array([0.19, 0.45, 0.29, 0.15, 0.24])
############################################

def corr_plot_PC(transformed_data, flux_sig_corr, color_coding = False, sv=False):
    """
    NAME:
    corr_plot_PC
    
    DESCRIPTION:
    UPDATE IT.....
    
    INPUT:
    transformed_data = 
    flux_sig_corr = 
    
    INPUT KEYWORD PARAMETERS:
    color_coding - You may want to color-code the correlation plots. 
    By default correlation plots are not color coded.
    sv- You may want to save the file.
    By default figure is not saved.
    
    """
    n0 = input('Principal Component for which correlation plots are to be generated:')
    n_PC = int(n0)-1
    if n_PC > 4 or n_PC<0:
        print ('Error! Principal Components can only go from 1 to 5')
        sys.exit()
    
    if color_coding == True:
        print ('============================================================================')
        print ('Enter 0 to color code the coorelation plots with PC1')
        print ('Enter 1 to color code the coorelation plots with PC2')
        print ('Enter 2 to color code the coorelation plots with PC3')
        print ('Enter 3 to color code the coorelation plots with PC4')
        print ('Enter 4 to color code the coorelation plots with PC5')
        print ('Enter 5 to color code the coorelation plots with PAH ratio 6.2/11.2')
        print ('Enter 6 to color code the coorelation plots with PAH ratio 7.7/11.2')
        print ('Enter 7 to color code the coorelation plots with PAH ratio 8.6/11.2')
        print ('Enter 8 to color code the coorelation plots with PAH ratio 11.0/11.2')
        print ('Enter 9 to color code the coorelation plots with PAH ratio 6.2/8.6')
        print ('Enter 10 to color code the coorelation plots with PAH ratio 7.7/8.6')
        print ('Enter 11 to color code the coorelation plots with PAH ratio 6.2/11.0')
        print ('Enter 12 to color code the coorelation plots with PAH ratio 7.7/11.0')
        print ('Enter 13 to color code the coorelation plots with PAH ratio 8.6/11.0')
        print ('Enter 14 to color code the coorelation plots with PAH ratio 6.2/7.7')
        print ('============================================================================')
    
        n_color = input('Enter value to color code the coorelation plots:')
        n_color = int(n_color)
        if n_color < 0 or n_color > 14 :
            print ('Error! Please enter value between 0 and 14')
            sys.exit()
        else:
            pass
            
    list_symbols_color = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    mapping_color = {'0' : transformed_data[:,0], 
           '1': transformed_data[:,1],
           '2': transformed_data[:,2],
           '3': transformed_data[:,3],
           '4': transformed_data[:,4],
           '5': (flux_sig_corr[:,0]/flux_sig_corr[:,1]),
           '6': (flux_sig_corr[:,2]/flux_sig_corr[:,1]),
           '7': (flux_sig_corr[:,4]/flux_sig_corr[:,1]),
           '8': (flux_sig_corr[:,3]/flux_sig_corr[:,1]),
           '9': (flux_sig_corr[:,0]/flux_sig_corr[:,4]),
           '10': (flux_sig_corr[:,2]/flux_sig_corr[:,4]),
           '11': (flux_sig_corr[:,0]/flux_sig_corr[:,3]),
           '12': (flux_sig_corr[:,2]/flux_sig_corr[:,3]),
           '13': (flux_sig_corr[:,4]/flux_sig_corr[:,3]),
           '14': (flux_sig_corr[:,0]/flux_sig_corr[:,2])} 
    mapping_color_labels = {'0' : r'$PC_{1}$',
                   '1': r'$PC_{2}$',
                   '2': r'$PC_{3}$',
                   '3': r'$PC_{4}$', 
                   '4': r'$PC_{5}$', 
                   '5': r' 6.2/11.2',
                   '6': r' 7.7/11.2',
                   '7': r' 8.6/11.2',
                   '8': r' 11.0/11.2',
                   '9': r' 6.2/8.6',
                   '10': r' 7.7/8.6',
                   '11': r' 6.2/11.0',
                   '12': r' 7.7/11.0',
                   '13': r' 8.6/11.0',
                   '14': r' 6.2/7.7'}
       
        
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(34, 34))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    
    PC = transformed_data[:,n_PC]
    if color_coding == True:
        cm = plt.cm.get_cmap('viridis')
        z = mapping_color[list_symbols_color[n_color]]
        images = []
    else:
        pass
   
    
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
            
            plot_data = mapping[list_symbols[n]]
            if color_coding == True:
                images.append(ax[i,j].scatter(PC, plot_data, c=z, cmap= cm))
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
                
            else:
                ax[i,j].plot(PC, plot_data, color='royalblue', marker = 'o', markersize=4, linestyle='None')
                r_value, p_value = ss.pearsonr(PC, plot_data)
                corr_coeff = round((r_value**(2)), 4)
                if n < 6:
                    ax[i,j].text(0.39*max(PC), 1.05*min(plot_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
                elif n > 5 and n < 10:
                    ax[i,j].text(0.39*max(PC), 0.90*max(plot_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
                else:
                    ax[i,j].text(0.39*max(PC), 1.05*min(plot_data), r"$R^{2}$: "+ str(corr_coeff), color='black', alpha=0.75, fontsize=27)
      
            ax[i,j].set_ylabel(mapping_ylabels[list_symbols[n]],fontsize=27)
            ax[i,j].set_xlabel(mapping_labels_PCs[list_symbols_PCs[n_PC]], fontsize=27)  
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
            
    if color_coding == True:
        cb = plt.colorbar(images[0], ax=ax.ravel().tolist(), orientation = 'horizontal', pad = 0.035)
        cb.ax.tick_params(labelsize=27)
        cb.ax.set_xlabel(mapping_color_labels[list_symbols_color[n_color]], fontsize = 27)
    else:
        pass
    
    if sv==True:
        plt.savefig('correlation_PC' + str(n_PC+1) + 'ngc2023' + '.eps', bbox_inches='tight')
    else:
        pass
    
    plt.show()
    
    return
############################################
flux_G76_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (7))
flux_G78_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (8))
ratio_south = flux_G78_south/flux_G76_south
Go_Stock_south = 10**((1.70-ratio_south)/0.28)
flux_G76_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (7))
flux_G78_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (6))
ratio_north = flux_G78_north/flux_G76_north
Go_Stock_north = 10**((1.70-ratio_north)/0.28)

Go_Stock = np.concatenate((Go_Stock_south, Go_Stock_north))

############################################
PAHTAT_south = np.loadtxt("ngc_2023_data_south.csv", delimiter = ',', skiprows = 1, usecols = (6))
Go_PAHTAT_south = 10**((PAHTAT_south-1.21)/(-0.23))
PAHTAT_north = np.loadtxt("ngc_2023_data_north.csv", delimiter = ',', skiprows = 1, usecols = (5)) 
Go_PAHTAT_north = 10**((PAHTAT_north-1.21)/(-0.23))

Go_PAHTAT = np.concatenate((Go_PAHTAT_south, Go_PAHTAT_north))
############################################
def spatial_maps(sv=False):
    
    """
    NAME:
    spatial_maps
    
    DESCRIPTION:
    UPDATE IT.....
    
    INPUT KEYWORD PARAMETERS: 
    sv- You may want to save the file.
    By default figure is not saved.
    
    """
    
    
    
    print ('============================================================================')
    print ('Enter 0 if you want to generate spatial map of PC1')
    print ('Enter 1 if you want to generate spatial map of PC2')
    print ('Enter 2 if you want to generate spatial map of PC3')
    print ('Enter 3 if you want to generate spatial map of PC4')
    print ('Enter 4 if you want to generate spatial map of PC5')
    print ('Enter 5 if you want to generate spatial map of 6.2 micron flux')
    print ('Enter 6 if you want to generate spatial map of 7.7 micron flux')
    print ('Enter 7 if you want to generate spatial map of 8.6 micron flux')
    print ('Enter 8 if you want to generate spatial map of 11.0 micron flux')
    print ('Enter 9 if you want to generate spatial map of 11.2 micron flux')
    print ('Enter 10 if you want to generate spatial map of G_0 - Stock and Peeters 2017')
    print ('Enter 11 if you want to generate spatial map of G_0 - Pilleri et al 2012')
    print ('============================================================================')
    
    n = input('Enter value corresponding to spatial map you want to generate:')
    n = int(n)
    if n < 0 or n > 11 :
        print ('Error! Please enter value between 0 and 11')
        sys.exit()
        
    list_symbols_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    mapping_map = {'0' : transformed_data[:,0], 
           '1': transformed_data[:,1],
           '2': transformed_data[:,2],
           '3': transformed_data[:,3],
           '4': transformed_data[:,4],
           '5': flux[:,0],
           '6': flux[:,2],
           '7': flux[:,4],
           '8': flux[:,3],
           '9': flux[:,1],
           '10': Go_Stock,
           '11': Go_PAHTAT}

    plot_map = mapping_map[list_symbols_map[n]]
    ind_sigma_corr = np.where(ind_sigma_flag_array==0)
    
    
    if n < 5:
        for i in range (0, len(ind_sigma_corr[0])):
            a = np.insert(plot_map, ind_sigma_corr[0][i], 0)
            plot_map = a
    else:
        pass
    
    plot_map_south = plot_map[0:1972]
    plot_map_north = plot_map[1972:2572]
    plot_map_south = np.reshape(plot_map_south, (34, 58))
    plot_map_north = np.reshape(plot_map_north, (20, 30))


    cont_112_south = flux_south[:,1]
    cont_77_south = flux_south[:,2]
    cont_112_north = flux_north[:,1]
    cont_77_north = flux_north[:,2]
    cont_112_south = np.reshape(cont_112_south, (34, 58))
    cont_77_south = np.reshape(cont_77_south, (34, 58))
    cont_112_north = np.reshape(cont_112_north, (20, 30))
    cont_77_north = np.reshape(cont_77_north, (20, 30))
    
    mask_array_south = ind_sigma_flag_array[0:1972]
    mask_array_north = ind_sigma_flag_array[1972:2572]
    
    if n == 10:
        mask_array_south = np.where(flux_G76_south == 0, 0, mask_array_south)
        mask_array_north = np.where(flux_G76_north == 0, 0, mask_array_north)
    elif n== 11:
        mask_array_south = np.where(PAHTAT_south == 0, 0, mask_array_south)
        mask_array_north = np.where(PAHTAT_north == 0, 0, mask_array_north)
    else:
        pass
    
   
    mask_array_south = np.reshape(mask_array_south, (34, 58))
    mask_array_south = mask_array_south.astype(np.float)
    mask_array_north = np.reshape(mask_array_north, (20, 30))
    mask_array_north = mask_array_north.astype(np.float)
    
    if n == 10 or n == 11:
        mask_array_south[26:34,51:58] = 0
    else:
        pass

        
    plot_map_south = plot_map_south.astype(np.float)
    plot_map_south[np.where(mask_array_south == 0)] = np.nan
    mask_array_south[np.where(mask_array_south != 0)] = np.nan
    plot_map_north = plot_map_north.astype(np.float)
    plot_map_north[np.where(mask_array_north == 0)] = np.nan
    mask_array_north[np.where(mask_array_north != 0)] = np.nan
    
    plot_map_south = plot_map_south.T
    cont_112_south = cont_112_south.T
    cont_77_south = cont_77_south.T
    mask_array_south = mask_array_south.T
    plot_map_north = plot_map_north.T
    cont_112_north = cont_112_north.T
    cont_77_north = cont_77_north.T
    mask_array_north = mask_array_north.T
 
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')
    fig.subplots_adjust(wspace=1.0)
    
    
    ax[0].imshow(mask_array_south, cmap='gray', vmin=0,vmax=1)
    if n == 10 or n == 11:
        im = ax[0].imshow(plot_map_south, cmap='viridis', norm = colors.LogNorm())
    else:
        im = ax[0].imshow(plot_map_south, cmap='viridis')
    cb = fig.colorbar(im, ax=ax[0], shrink = 0.5)
    cb.ax.tick_params(labelsize=18)
    ax[0].contour(cont_112_south, levels = [0.00000366, 0.00000464, 0.00000564, 0.00000678], colors=('k',),linestyles=('solid'))
    ax[0].contour(cont_77_south, levels = [0.0000140, 0.0000156, 0.0000170, 0.0000190], colors=('white',),linestyles=('solid'))
    ax[0].annotate(r"$S^{'}$", xy=(3, 2), xytext=(4, 7), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=25)
    ax[0].annotate('SE', xy=(1, 14), xytext=(4, 9), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=25)
    ax[0].annotate('SSE', xy=(2, 28), xytext=(0.8, 35), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=25)
    ax[0].annotate('S', xy=(16, 31), xytext=(25, 21), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=25)
    ax[0].set_xlabel('Pixel Number', fontsize=18)
    ax[0].set_ylabel('Pixel Number', fontsize= 18)
    ax[0].tick_params(direction='out', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize = 18)
    ax[0].tick_params(direction='out', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)
    majorLocator = MultipleLocator(10)
    ax[0].xaxis.set_major_locator(majorLocator)
    ax[0].yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(1.0)
    ax[0].xaxis.set_minor_locator(minorLocator)
    ax[0].yaxis.set_minor_locator(minorLocator)
    
    ax[1].imshow(mask_array_north, cmap='gray', vmin=0,vmax=1)
    if n == 10 or n == 11:
        im = ax[1].imshow(plot_map_north, cmap='viridis', norm = colors.LogNorm())
    else:
        im = ax[1].imshow(plot_map_north, cmap='viridis')
    cb = fig.colorbar(im, ax=ax[1], shrink = 0.5)
    cb.ax.tick_params(labelsize=18)
    ax[1].contour(cont_112_north, levels = [0.00000142, 0.00000157,0.00000175, 0.00000199], colors=('k',),linestyles=('solid'))
    ax[1].contour(cont_77_north, levels = [0.00000554, 0.00000630,0.00000680, 0.00000720], colors=('white',),linestyles=('solid'))
    ax[1].annotate('NW', xy=(11.7, 18.1), xytext=(8.3, 22), color = 'white', arrowprops=dict(facecolor='white', shrink=0.05), fontsize=18)
    ax[1].annotate('N', xy=(7.7, 7.9), xytext=(7.9, 4.1), color = 'white', arrowprops=dict(facecolor='white', shrink=0.05), fontsize=18)
    ax[1].set_xlabel('Pixel Number', fontsize=18)
    ax[1].set_ylabel('Pixel Number', fontsize= 18)
    ax[1].tick_params(direction='out', which = 'major', bottom= True, left= True, top=True, right=True, length = 12, labelsize = 18)
    ax[1].tick_params(direction='out', which = 'minor', bottom= True, left= True, top=True, right=True, length = 6)
    majorLocator = MultipleLocator(5)
    ax[1].xaxis.set_major_locator(majorLocator)
    ax[1].yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.5)
    ax[1].xaxis.set_minor_locator(minorLocator)
    ax[1].yaxis.set_minor_locator(minorLocator)
    
    if sv==True:
        plt.savefig('spatial_map_ngc2023.eps', bbox_inches='tight')
    else:
        pass
    
    
    plt.show()
    
    return
############################################
                   
