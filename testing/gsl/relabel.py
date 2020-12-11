import numpy as np
import pandas as pd
import sys
import os

if os.path.isdir(r'D:\OneDrive - Vrije Universiteit Brussel'):
    os.chdir(r'D:\OneDrive - Vrije Universiteit Brussel')
elif os.path.isdir(r'C:\Users\Frederik Priem\OneDrive - Vrije Universiteit Brussel'):
    os.chdir(r'C:\Users\Frederik Priem\OneDrive - Vrije Universiteit Brussel')
elif os.path.isdir(r'C:\OneDrive\OneDrive - Vrije Universiteit Brussel'):
    os.chdir(r'C:\OneDrive\OneDrive - Vrije Universiteit Brussel')
wd = os.getcwd()

sys.path.append(wd + r'\desktop\GENLIB\Work_packages\WP2_GUL_tools')

from gultools.io import *


lut_path = r'D:\OneDrive - Vrije Universiteit Brussel\desktop\GENLIB\Work_packages\WP3_POC\0_GUL\classification_system\lut_relabel.csv'
lib1_path = r'D:\OneDrive - Vrije Universiteit Brussel\desktop\GENLIB\Work_packages\WP2_GUL_tools\testing\gsl\ECOSTRESS_bc_level1.sli'
lib2_path = r'D:\OneDrive - Vrije Universiteit Brussel\desktop\GENLIB\Work_packages\WP2_GUL_tools\testing\gsl\DLR_bc_level1.sli'

lut = pd.read_csv(lut_path)
print(lut)

spectra, metadata = read_envi_library(lib1_path)
labels = metadata['spectra names'].astype(float)
labels = labels.astype(int)

new_labels = np.empty(labels.size, dtype='U100')

for label in np.unique(labels):

    if label in lut['lc1_id'].values:

        ind = np.where(lut['lc1_id'].values == label)[0][0]
        new_label = lut['lc1'].values[ind]
        new_labels[np.where(labels == label)[0]] = new_label

print(new_labels)
