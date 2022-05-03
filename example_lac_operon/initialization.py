""" initialization script, import and copy data to like named variables in other modules.
"""

##############################
### Integration parameters ###
##############################

# ranges over which parmeters in classes_eds2 classes can vary (float needed). Defined in mutation.py

T = 1.0 #typical time scale
C = 1.0 #typical concentration
L = 1.0 #typical size for diffusion

# dictionary key is the Class.attribute whose range is given as a real number or as a LIST
# indicating the interval over which it has to vary

dictionary_ranges={}
dictionary_ranges['Species.degradation']=1.0/T
#dictionary_ranges['Species.concentration']=C   # for certain species, eg kinases that are not trans
dictionary_ranges['Species.diffusion']=L   # for ligands diffusion
dictionary_ranges['TModule.rate']=C/T
dictionary_ranges['TModule.basal']=0.0
dictionary_ranges['CorePromoter.delay']=0   # convert to int(T/dt) in run_evolution.py
dictionary_ranges['TFHill.hill']=5.0
dictionary_ranges['TFHill.threshold']=C
dictionary_ranges['PPI.association']=1.0/(C*T)
dictionary_ranges['PPI.disassociation']=1.0/T
dictionary_ranges['Phosphorylation.rate']=1.0/T
dictionary_ranges['Phosphorylation.hill']=5.0
dictionary_ranges['Phosphorylation.threshold']=C
dictionary_ranges['Phosphorylation.dephosphorylation']=1.0/T
dictionary_ranges['Degradation.rate']=1.0/T

# when key='relative_variation' is present, use this fraction of parameter range listed
# above, to vary parameter for each mutation.  Comment out line to pick parameter
# randomly from range defined above.
# dictionary_ranges['relative_variation']=0.1

#################################################################################
# names of c-code files needed by deriv2.py (. for extension only)
# normally need 'header', 'utilities' (loaded before python derived routines and
# ['fitness', 'geometry', 'init_history', 'input', 'integrator', 'main' ] placed afterwards
# skip by setting cfile[] = ' ' or ''

cfile = {}
cfile['fitness'] = 'fitness.c'
cfile['init_history'] = 'init_history.c'
cfile['input'] =  'input.c'

pfile = {}

####################################
### Mutation rates for evolution ###
####################################
dictionary_mutation={}

# Rates for nodes to add
dictionary_mutation['random_gene()']=0.01
dictionary_mutation['random_gene(\'TF\')']=0.01
dictionary_mutation['random_gene(\'Kinase\')']=0.00
dictionary_mutation['random_gene(\'Ligand\')']=0.00
dictionary_mutation['random_gene(\'Receptor\')']=0.00
dictionary_mutation['random_Interaction(\'TFHill\')']=0.005
dictionary_mutation['random_Interaction(\'PPI\')']=0.005
dictionary_mutation['random_Interaction(\'Phosphorylation\')']=0.000
dictionary_mutation['random_Interaction(\'Degradation\')']=0.00

# Rates for nodes to remove
dictionary_mutation['remove_Interaction(\'TFHill\')']=0.015
dictionary_mutation['remove_Interaction(\'PPI\')']=0.015
dictionary_mutation['remove_Interaction(\'CorePromoter\')']=0.015
dictionary_mutation['remove_Interaction(\'Phosphorylation\')']=0.00
dictionary_mutation['remove_Interaction(\'Degradation\')']=0.00

# Rates to change parameters for a node
dictionary_mutation['mutate_Node(\'Species\')']=0.1
dictionary_mutation['mutate_Node(\'TFHill\')']=0.1
dictionary_mutation['mutate_Node(\'CorePromoter\')']=0.1
dictionary_mutation['mutate_Node(\'TModule\')']=0.1
dictionary_mutation['mutate_Node(\'PPI\')']=0.1
dictionary_mutation['mutate_Node(\'Phosphorylation\')']=0.1
dictionary_mutation['mutate_Node(\'Degradation\')']=0.1

#rates to change output tags.  See list_types_output array below
dictionary_mutation['random_add_output()']=0.0
dictionary_mutation['random_remove_output()']=0.0
dictionary_mutation['random_change_output()']=0.0 #NO OUTPUT CHANGE

#############################################################################
# parameters in various modules, created as one dict, so that can then be passed as argument

# Needed in deriv2.py to create C program from network
prmt = {}
prmt['nstep'] = 5000         #number of time steps
prmt['ncelltot'] = 1            #number of cells in an organism
prmt['nneighbor'] = 3 # must be >0, whatever geometry requires, even for ncelltot=1
prmt['ntries'] = 2   # number of initial conditions tried in C programs
prmt['dt'] = 0.05     # time step

# Generic parameters, transmitted to C as list or dictionary.
# dict encoded in C as: #define KEY value (KEY converted to CAPS)
# list encoded in C as: static double free_prmt[] = {}
# The dict is more readable, but no way to test in C if given key supplied.  So
# always include in C: #define NFREE_PRMT int. if(NFREE_PRMT) then use free_prmt[n]
# prmt['free_prmt'] = { 'step_egf_off':20 }  # beware of int vs double type
# prmt['free_prmt'] = [1,2]

# Needed in evolution_gill to define evol algorithm and create initial network
prmt['npopulation'] = 100
prmt['ngeneration'] = 1001
prmt['tgeneration'] = 1.0       #initial generation time (for gillespie), reset during evolution
prmt['noutput']=1    # to define initial network
prmt['ninput']=1
prmt['freq_plot'] = 10  #plot time course every generation%freq_plot = 0
prmt['freq_stat'] = 5     # print stats every freq_stat generations
prmt['frac_mutate'] = 0.5 #fraction of networks to mutate
prmt['redo'] = 1   # rerun the networks that do not change to compute fitness for different IC

# used in run_evolution,
prmt['nseed'] = 25   # number of times entire evol procedure repeated, see main program.
prmt['firstseed'] = 0  #first seed

# multipro_level used in evolution_gillespie.  =2 if using parallel processing(pypar) =1 for threading =0 for serial processing

prmt['multipro_level']=1
prmt['plot']=0
prmt['langevin_noise']=0

#####################
### PARETO PARAM. ###
#####################
prmt['pareto']=0 # a flag, set to 1 to run pareto  
prmt['npareto_functions']= 1 # number of functions to use for pareto optimization.  
prmt['rshare']= 0.0



# To restart from a Restart_* file created in Seed* directories:
#    Move the Restart file into a new model directory (see run_evolution -m option)
#    Adjust the entries below as desired and run_evolution module.
#    Note option to either exactly reproduce prior evolution simulation or just initialize with some
# prior population and then do indepedent evolution

# The parameters for restart related functions, grouped into subdictionary, used in evolution_gillespie.py
# The flag to restart from file is file name, None skips restart file and uses new instances of init functions
# If restart the numbering of generations continuous with that of restart file.
# To redo a saved population with different random number, use ['same_seed'] = False
# Note the 'freq' parameter below, if ~= npopulation then Restart file size ~1/2 Bests file
prmt['restart'] = {}
prmt['restart']['activated'] = False #indicate if you want to restart or not
prmt['restart']['dir'] =  "Exponential/Seed0" # the directory of the population you want to restart from
prmt['restart']['kgeneration'] = 50  # restart from After this generation number (see loop in Population.evolution)
prmt['restart']['freq'] = 50  # save population every freq generations
prmt['restart']['same_seed'] = True  # get seed of random() from restart file to reproduce prior data.


list_unremovable=['Input']
# control the types of output species, default 'Species' set in classes_eds2.  This list targets
# effects of dictionary_mutation[random*output] events to certain types, when we let output tags
# move around

list_types_output=['Species','TF']

# necessary imports to define following functions
import random
from phievo.Networks import mutation

# Two optional functions to add input species or output genes, with IO index starting from 0.
# Will overwrite default variants in evol_gillespie if supplied below

def init_network():
    seed=int(random.random()*100000)
    g=random.Random(seed)
    L=mutation.Mutable_Network(g)
    L.fixed_activity_for_TF = 0

    parameters=[['Degradable', mutation.sample_dictionary_ranges('Species.degradation',random) ]]
    parameters.append(['Input',0])
    parameters.append(['Complexable'])
    TF1=L.new_Species(parameters)

    parameters=[['Degradable', mutation.sample_dictionary_ranges('Species.degradation',random) ]]
    parameters.append(['Input',1])
    parameters.append(['Complexable'])
    TF2=L.new_Species(parameters)

    [tm, prom, o1] = L.random_gene('TF')
    o1.mutable = False
    o1.add_type(['Output',0])
    o1.clean_type('Complexable')
    L.write_id()
    
    return L

def fitness_treatment(population):
    # Function to slightly change the fitness of the networks
    for ind in population:
        try:
            ind.fitness += 0.001*random.random()
        except Exception:
            ind.fitness = None

