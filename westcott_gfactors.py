# Calculate Westcott g factors for nuclei whose thermal-neutron capture cross sections require a correction to the "1/v" rule due to the presence of low-energy resonances.

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import glob
import re
import csv


# Constants and basic functions
v_0 = 2200 #thermal-neutron velocity, m/s
eV = 1.602189e-19 #J
kB = 1.38066e-23 #Boltzmann's constant, J/K
m_n = 1.00866501 *1.660566e-27 #neutron mass, kg
E_0 = 1/2 * m_n * v_0**2 / eV #eV

# Convert neutron energy (eV) to velocity (m/s)
def vel(E):  
    E_joules = E*eV
    return np.sqrt(2*E_joules/m_n)

# Maxwellian velocity distribution at a given temperature T (K)
def phi_Maxwellian(T, v_array):  
    phi = []
    vt = np.sqrt(2*kB*T/m_n)
    for v in v_array:
        phi.append(2 * np.exp(-v**2/vt**2) * v**3/vt**4)
    return np.array(phi)


# Evaluate g-factor using irregularity function method described by Molnar, et al. (Eq. 1-5)
def gw_irregularity(E_resonance, Gamma, vn=np.linspace(1,100000,100000), phi=phi_Maxwellian(293,np.linspace(1,100000,100000)), T=0.01):
    # Lorentzian lineshape for the irregularity function, per Molnar eqn. 1-3
    def del_0(v, E_r, G):
        E = 1/2 * m_n * v**2 / eV #eV
        del_0 = ((E_resonance - E_0)**2 + Gamma**2/4)/((E_resonance - E)**2 + Gamma**2/4)
        return del_0

    if T>0.01:
        # Use Maxwellian spectrum, otherwise use arbitrary (user-defined) neutron spectrum, in which case T is integrated out in normalization
        phi = phi_Maxwellian(T, vn)

    # Neutron density function (Molnar Ch. 1, Table 1)
    def p(vn, phi, T):
        p_array = []
        for i in range(len(vn)):
            vt = np.sqrt(2 * kB * T / m_n)
            p_array.append(2 * vt * phi[i] / (np.sqrt(np.pi) * vn[i]))
        N = trapezoid(np.array(p_array), vn)  # Normalization factor, to ensure integral of p(T,v) integrates to unity (Molnar p. 12)
        return np.array(p_array)/N

    d = []
    for i in range(len(vn)):
        d.append(del_0(vn[i], E_resonance, Gamma))
    d = np.array(d)
    return trapezoid(d * p(vn, phi, T), vn)



# Pull capture cross sections from ENDF
def sigma_ENDF(capture_data_path, tgt):
    # Load neutron-capture cross-section data
    capture_list = [x for x in glob.glob("{0}/*.csv".format(capture_data_path))]
    capture_dict = {}
    
    for c in capture_list:
        c_file = c.split('capture_data/')[1]
        target = c_file.split('n-capture-')[1].split('.csv')[0]
        capture_dict.update({target: c_file})
    capture_dict = dict(sorted(capture_dict.items()))
    #for k,v in capture_dict.items(): print(k,v)

    def find_targets():
        return [target for (target, value) in capture_dict.items()]

    def get_MT102(target):
        df = None
        for k,v in capture_dict.items():
            if k==target:
                df = pd.read_csv("{0}/{1}".format(capture_data_path, v))
        if df is None:
            print("No capture-gamma cross section data for target nucleus: {0}".format(target))
            return
        else:
            return df

    targs = find_targets()
    df = get_MT102(tgt)

    # Convert to numpy arrays for interpolation and integration
    sigma = df.to_numpy()
    En = sigma[:,0]
    sigma_E = sigma[:,1]

    return(En, sigma_E)



# Import arbitrary neutron energy spectrum in csv format
def import_spectrum(csv_filename):
    with open(csv_filename,mode='r', encoding='utf-8-sig') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)  # Skip header line
        En = []
        dndE = []
        for lines in csvFile:
            En.append(float(lines[0]))
            dndE.append(float(lines[1])) 
    return(np.array(En), np.array(dndE))



# Integrate to evaluate Westcott g-factor for a Maxwellian neutron distribution
def gw_Maxwellian(T, E, sigma):
    v = vel(E)  #convert neutron energy to velocity
    dndv = phi_Maxwellian(T, v)
    sigma0 = np.interp(v_0, v, sigma)  #thermal cross section, barns
    return 1/(sigma0 * v_0) * trapezoid(dndv * v * sigma, v) / trapezoid(dndv, v)



# Integrate to evaluate Westcott g-factor for an arbitrary neutron distribution
def gw_arbitrary(E_spectrum, dndE_spectrum, E_sigma, sigma):
    v_sigma = vel(E_sigma)
    v_spectrum = vel(E_spectrum)
    dndv = dndE_spectrum
    
    # Interpolate to define flux everywhere the cross section is defined, with zeros outside range to integrate properly
    dndv_interp = np.interp(v_sigma, v_spectrum, dndv, left=0, right=0)
    
    sigma0 = np.interp(v_0, v_sigma, sigma)  #thermal cross section, barns

    return 1/(sigma0 * v_0) * trapezoid(dndv_interp * v_sigma * sigma, v_sigma) / trapezoid(dndv_interp, v_sigma)



#################
# MAIN SEQUENCE #
#################
if __name__ == '__main__':
    # Define target isotopes to test methods
    for test_isotope in ['Kr83','Eu151','Gd157','Re187']:
        print('\n',test_isotope)

        # Get neutron-capture cross section
        capture_data_PATH = "/Users/davidmatters/westcott/n-capture-gnds/capture_data"
        sigma_x, sigma_y = sigma_ENDF(capture_data_PATH, test_isotope)

        # Evaluate g-factor (Maxwellian distribution, 293 K)
        print("Maxwellian distribution, T=293 K: g_w = {0:.3f}".format(gw_Maxwellian(293, sigma_x, sigma_y)))

        # Evaluate g-factor (BRR thermal spectrum, 2002)
        En_BRR_thermal, dndE_BRR_thermal = import_spectrum('/Users/davidmatters/westcott/spectra/csv/bnc_thermal_spectrum_2002.csv')
        print("BRR thermal-neutron spectrum (293 K): g_w = {0:.3f}".format(gw_arbitrary(En_BRR_thermal, dndE_BRR_thermal, sigma_x, sigma_y)))

        # Evaluate g-factor (Maxwellian distribution, 140 K)
        print("Maxwellian distribution, T=140 K: g_w = {0:.3f}".format(gw_Maxwellian(140, sigma_x, sigma_y)))

        # Evaluate g-factor (BRR cold spectrum, 2012)
        En_BRR_cold, dndE_BRR_cold = import_spectrum('/Users/davidmatters/westcott/spectra/csv/bnc_cold_spectrum_2012.csv')
        print("BRR cold-neutron spectrum (140 K): g_w = {0:.3f}".format(gw_arbitrary(En_BRR_cold, dndE_BRR_cold, sigma_x, sigma_y)))

        # Evaluate g-factor (Maxwellian distribution, 21 K)
        print("Maxwellian distribution, T=21 K: g_w = {0:.3f}".format(gw_Maxwellian(21, sigma_x, sigma_y)))

        # Evaluate g-factor (FRM-II cold spectrum, 2008)
        En_FRM_cold, dndE_FRM_cold = import_spectrum('/Users/davidmatters/westcott/spectra/csv/frm-ii_cold_spectrum_2008.csv')
        print("FRM-II cold-neutron spectrum (21 K): g_w = {0:.3f}".format(gw_arbitrary(En_FRM_cold, dndE_FRM_cold, sigma_x, sigma_y)))

        #Lowest-energy resonance parameters for select nuclei from ENDF/B-VIII.1 and Mughabghab's Atlas, to demonstrate irregularity method
        class Resonance:
            def __init__(self, isotope, E_resonance, Gamma):
                self.isotope = isotope
                self.energy = E_resonance
                self.width = Gamma

        resonances_ENDF = [
            Resonance('Kr83', -9.81, 3.8400000e-01),
            Resonance('Eu151', -0.0609, 0.1052768),
            Resonance('Gd157', 0.0314, 0.1072 + 4.74E-04),
            Resonance('Re187', -4.03, 7.4371000e-02),
            ]

        resonances_Atlas = [
            Resonance('Kr83', -9.81, 252/1000),
            Resonance('Eu151', -0.00362, 95.8/1000),
            Resonance('Gd157', 0.0314, 107/1000),
            Resonance('Re187', -3.94, 57.8/1000)
            ]

        def get_ENDF_parameters(isotope):
            for resonance in resonances_ENDF:
                if resonance.isotope == isotope:
                    return(resonance.energy, resonance.width)

        Er_test, Gamma_test = get_ENDF_parameters(test_isotope)

        # Evaluate g-factor using irregularity method (Maxwellian distribution, 293 K)
        print("Irregularity method, Maxwellian distribution w/T=293 K: g_w = {0:.3f}".format(gw_irregularity(Er_test, Gamma_test, np.linspace(1,100000,100000), phi_Maxwellian(293,np.linspace(1,100000,100000)), 293)))

        # Evaluate g-factor using irregularity method (BRR cold spectrum, 2012)
        print("Irregularity method, BRR cold-neutron spectrum (140 K): g_w = {0:.3f}".format(gw_irregularity(Er_test, Gamma_test, vel(En_BRR_cold), dndE_BRR_cold)))

        # Evaluate g-factor using irregularity method (BRR thermal spectrum, 2002)
        print("Irregularity method, BRR thermal-neutron spectrum (293 K): g_w = {0:.3f}".format(gw_irregularity(Er_test, Gamma_test, vel(En_BRR_thermal), dndE_BRR_thermal)))

        # Evaluate g-factor using irregularity method (FRM-II cold spectrum, 2008)
        print("Irregularity method, FRM-II cold-neutron spectrum (21 K): g_w = {0:.3f}".format(gw_irregularity(Er_test, Gamma_test, vel(En_FRM_cold), dndE_FRM_cold)))
    
