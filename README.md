# Principal Component Analysis of PAH fluxes in reflection nebula NGC 2023
We performed the principal component analysis (PCA) of the polycyclic aromatic hydrocarbon (PAH) emission features at 6.2, 7.7, 8.6, 11.0 and 11.2 micron in the reflection nebula NGC 2023 to study the previously reported variations in the PAH emission. Fluxes of the emission features are taken from Peeters et al. 2017. 

PCA is a statistical technique for data visualization and dimensionality reduction that transforms a set of correlated variables in a given data set into a new set of uncorrelated variables called the principal components (PCs) using an orthogonal transformation.

The 'pca' folder contains the data and the python code to perform the PCA analysis on the PAH fluxes of NGC 2023.

To analyse the data, following figures are made using the code:

Biplots - Projection of original variables in the reference frame of PCs.

Eigen Spectra of PCs - Spectrum representative of the eigen vector associated with a PC

Characteristic Spectra of PCs - An artificial PAH spectrum corresponding to a PC

Correlation plots of PCs with various PAH fluxes

Spatial maps of PCs, radiation field strengths, and PAH fluxes

# Following packages are required to run the code:
numpy

math

scipy

matplotlib

sklearn


# Type the following to run the code:
python pca.py

Note: Uncomment lines at the end of the code, depending on the figure you wish to create.




