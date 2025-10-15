# Extract neutron-capture cross sections from GNDS-formatted ENDF files.  Requires installation of FUDGE (https://github.com/LLNL/fudge)

from fudge import map as fudgeMap
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import glob
import re


# Read in FUDGE map of ENDF/B-8.1 data and return reaction suite
def read_fudgeMap(fudgeMap_path, projectile_type, target_nucleus):
    ENDF81 = fudgeMap.read(fudgeMap_path)
    rxn_suite = ENDF81.find(projectile=projectile_type, target=target_nucleus).read()
    return(rxn_suite)


# Create target list from FUDGE map and ENDF XML files
def create_targetList(ENDF_XML_path):
    # Create list of neutron targets from XML ENDF files
    ENDF_XML_PATH = ENDF_XML_path
    xml_list = [x for x in glob.glob("%s/*.xml"%ENDF_XML_path)]
    n_targets = []
    for xml in xml_list:
        xml_file = xml.split('neutrons/')[1]
        Z = int(xml_file.split('n-')[1].split('_')[0])
        element = xml_file.split('_')[1]
        A = None
        try:
            A = int(xml_file.split('_')[2].split('.')[0])
        except ValueError:
            A = xml_file.split('_')[2].split('.')[0]
            if 'm' in A:
                if A[0]=='0' and A[1]=='0':
                    A = A.strip(A[0:2])
                elif A[0]=='0':
                    A = A.strip(A[0:1])
                else:
                    pass
                A = A.replace('m','_m')

        n_targets.append(element+str(A))
    n_targets = sorted(n_targets)
    print("List object contains {0} n-reaction targets from XML-formatted ENDF libraries.\n".format(len(n_targets)))
    print(n_targets)
    return n_targets


# Define target directory to write and store capture-gamma cross-section CSV data. 
def create_sigma_csv(n_targets, fudgeMap_path, capture_data_path):
    # Define target directory for CSV files to go
    TARGETDIR_CAPTURE = capture_data_path
    columns = ['energy [eV]', 'cross section [b]']

    # Read in target data and write CSV files, ignoring n+n1 and (for now) n+U238 due to parsing problems
    for t in n_targets:
        if t!='n1' and t!='U238':
            RS = read_fudgeMap(fudgeMap_path,'n',t)
            cs_data = None

            # Do some regex on target label
            letters_pattern = r'\D+'
            numbers_pattern = r'\d+'
            chem_symbol = str(re.findall(letters_pattern, t)[0])
            target_mass = int(re.findall(numbers_pattern, t)[0])
            residual_mass = target_mass+1
            residual = str(chem_symbol)+str(residual_mass)
            print(residual)

            for reaction in RS:
                if str('{0} + photon [inclusive]'.format(residual)) in reaction.label:
                    try:
                        crossSection = reaction.crossSection.evaluated.toPointwise_withLinearXYs(accuracy=1e-3, lowerEps=1e-8)
                        cs_data = crossSection
                    except:
                        print(f"Could not extract cross section for reaction '{reaction.label}'!")

            # Extract pointwise cross-section data from `cs_data` object into a new DataFrame:
            try:
                df_cs = pd.DataFrame(list(cs_data), columns=columns)
                df_cs.to_csv("{0}/n-capture-{1}.csv".format(TARGETDIR_CAPTURE,t), index=False)
            except TypeError:
                if cs_data is None:
                    print("Reaction suite for target {0}: No (n,g) reaction channel".format(t))
    return
      
    
# Get radiative thermal-neutron capture reactions from the RS and return as a pandas DataFrame
def get_capture_cs(RS):
    cs = None
    for reaction in RS:
        #(n,g) reactions are identified with '(A+1)Z + photon [inclusive]'
        if str('+ photon [inclusive]') in reaction.label:
            try:
                cs = reaction.crossSection.evaluated.toPointwise_withLinearXYs(accuracy=1e-3, lowerEps=1e-8)
                columns = ['energy [eV]', 'cross section [b]']
                df = pd.DataFrame(list(cs), columns=columns)
                return df
            except:
                print(f"Could not extract cross section for reaction '{reaction.label}'!")
                return



#################
# MAIN SEQUENCE #
#################
if __name__ == '__main__':
    # Data locations
    fudgeMap_PATH = '/Users/davidmatters/ENDF/ENDF-B-VIII.1-GNDS/neutrons.map'
    ENDF_XML_PATH = '/Users/davidmatters/ENDF/ENDF-B-VIII.1-GNDS/neutrons'
    capture_CSV_PATH = '/Users/davidmatters/westcott/n-capture-gnds/capture_data'

    # Test by displaying the reaction suite for a specific nucleus
    test_suite = read_fudgeMap(fudgeMap_PATH,'n','W186')
    print(test_suite.toString())
    
    # Get (n,g) cross section and plot
    cs_dataframe = get_capture_cs(test_suite)
    cs_data = cs_dataframe.to_numpy()
    plt.loglog(cs_data[:,0], cs_data[:,1])
    plt.xlabel(f"Incident energy ({test_suite.domainUnit})")
    plt.ylabel("Cross section (b)")
    plt.title(f"n + '{test_suite.target}' reaction")
    plt.show()

    # Create csv files of cross sections
    #test_targets = create_targetList(ENDF_XML_PATH)
    #create_sigma_csv(test_targets, fudgeMap_PATH, capture_CSV_PATH)
    
    
