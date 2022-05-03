"""
Defines the Class Population with her principal method, evolution, which
evolve a set of networks. All initialization done from an initialization.py
file. All the modules are initialized through run_evolution.py.

The initial networks to evolve, can be built from just the input/output genes,
a predefined newtork, or restarted from any saved population from a previous
run. (See initialization file for details)

The time between generations is variable, and about the same for all species,
we sample the mutation rates with a gillespie like algorithm, hence the name

The evolution method will write the following files in the namefolder given
as argument to Population.__init__ stdout basic info each generation:
* Bests = for generation, the network with best fitness in text form to edit or process with stat_best_net.py
* Restart* = binary dbm type file with data to restart evolution at selected generation numbers
* graphic files with time course and best network diagram at selected generations
"""
from phievo import __verbose__,__silent__

if __verbose__:
    print('Execute evolution_gillespie.py')

import phievo.Networks.classes_eds2 as classes_eds2
import phievo.Networks.mutation as mutation
import gc # Garbage collector
from math import log,sqrt
import copy,os,random,sys,glob
import shelve
import time, pickle, dbm  # for restart's
import phievo.Populations_Types.population_stat as pop_stat
from phievo import test_STOP_file
import re

import os
import sys
#added for selection
sys.path.insert(1,os.path.join(os.getcwd(),'phievo'))
sys.path.insert(1,os.path.join(os.getcwd(),'misc'))
from selection import *
#---

#########################
### Global Parameters ###
#########################

# read various parameters for ODE generation and evolution from input file into dictionary called prmt
prmt = {}

# dictionary of line labels and commands to define statistics extracted from networks see Class NetworkStat
stat_dict = {}
stat_dict['Node'] = lambda net: net.dict_types['Node']
stat_dict['Species'] = lambda net: net.dict_types['Species']

count_interactions = ['TFHill', 'PPI', 'Phosphorylation']
count_keys = list(mutation.dictionary_mutation.keys())
for i in count_interactions:
    rates0 = [mutation.dictionary_mutation[kk]==0 for kk in count_keys if kk.find(i) >= 0]
    
    if len(rates0) > 0:
        continue
    stat_dict[i] = lambda net: net.dict_types.get(i,None)

#######################
### Dummy Functions ###
#######################

def fitness_treatment(population):
    """default function for fitness treatment

    If necessary, should be implemented in the init*.py file
    """
    pass

def init_network(mutation):
    """Default function to create network

    It must be overwritten with function from the init*.py file
    otherwise stop the programm
    """
    raise NotImplementedError("must supply init_network() in initialization.py")

#########################
### General Functions ###
#########################
def restart(directory, generation, verbose = True):
    """Allow the user to restart an old run

        Args:
            directory (str): the directory of the restart file
            generation (int): the generation number

        Returns:
            rprmt (dict): the parameters of the run
            genus (list): the list of individuals(:class:`Network <phievo.Networks.mutation.Mutable_Network>`) of the population
    """

    restart_file = os.path.join(directory,"Restart_file")
    try:
        restart_data = shelve.open(restart_file,flag="r")
    except dbm.error:
        raise FileNotFoundError("The directory {0} does not have a restart file.".format(directory))
    if generation is None:
        generation = max([int(ss) for ss in restart_data.dict.keys()])

    try:
        rprmt, genus = restart_data[str(generation)]
    except KeyError:
        raise KeyError("Generation {0} is not stored in the restart file  for Seed{1}.".format(generation,prmt["restart"]["seed"]))
    restart_data.close()
    if verbose:
        print('successfully restarted from file= ', dir, 'generation= ', generation)
        print('header=', rprmt['header'])

    return rprmt,genus,generation

###################################
### Class Population Definition ###
###################################

class Population(selection_methods): #object removed for selesction

    """
    Define a population as a list of networks called Population.
    Genus and a principal method evolution.
    object means it is a newstyle class ! See the `web <https://wiki.python.org/moin/NewClassVsClassicClass>`_ for distinction between new and olds style class, important for inheritance

    Attributes:
        best_fitness (float): keep trace of the best fitness in the population
        genus (list): the list of individuals(:class:`Network <phievo.Networks.mutation.Mutable_Network>`) of the population
        same_seed (bool): indicate if the file is a restart or not
        tgeneration (float): starting hop time for the gillespie algorithm
        npopulation (int): size of te population
        bests_file (str): directory to save the data of evolution

    Main methods:
        evolution: launch the evolutionary algorithm
        pop_mutate_and_integrate: update the whole population
    """
    def __init__(self, namefolder):
        """ Copy a few parameters from prmt-dictionary that logically belong to the population, then
        setup various files and from initialization file decide how to initialize networks in genus.
        """

        #added for selection
        self.archive_log = []
        self.archive = []
        self.sparsests = []
        self.fitests = []
        #---


        self.best_fitness = sys.maxsize  # tag to determine if fitness increasing each generation
        self.best_fitness_counter = 0   # number of generations that best fitness has not changed.
        self.same_seed = False  # False -> starting from new data, reset to True below if replicating restart file
        self.generation0 = 0  # number of first generation, if replicating restart file set>0
        self.tgeneration = prmt['tgeneration']
        self.npopulation = prmt['npopulation']
        self.namefolder = namefolder   # directory where all data going
        self.n_mutations = 0 #number of mutations per generation

        #file to hold best network each generation
        self.data_file = os.path.join(namefolder,'data')
        self.bests_file = os.path.join(namefolder,'Bests_{}.net')

        # unique file name to save restart data,
        self.restart_file = os.path.join(namefolder,'Restart_file')

        # reset crucial parameters for loop over generations in evolution.
        # Set self.same_seed = True if want to exactly
        # recreate evolution that lead to restart data
        if prmt['restart']['activated']:
            rprmt, self.genus,prmt['restart']['kgeneration'] = restart(namefolder, prmt['restart'].get('kgeneration',None) )
            self.tgeneration = rprmt['tgeneration']
            if prmt['restart'].get('same_seed',True):
                random.setstate( rprmt['state'] )
                self.same_seed = True
            prmt['restart']['kgeneration'] +=1
            prmt['restart']['activated'] = False
            return None
        else:
            prmt['restart']['kgeneration'] = self.generation0
        # no restart, generate randomized list of networks from init file or routines supplied here.
        self.genus=[]
        for i in range(self.npopulation):
            L = init_network()
            L.write_id()
            self.genus.append(L)

    def __getitem__(self,index):
        """Allow the population to be indexed as a list"""
        return self.genus[index]

    def __len__(self):
        """Overload the population size"""
        return self.npopulation


    def storing(self,t_gen,net):
        """Store the work and various data for later analysis

        Network object are stored in individual pickle file in Seed{}/data
        Data are stored in a shelve called the Seed{}/Bests_{}.net

        Args:
            t_gen: the key (normally the generation number)
            net (:class:`Network <phievo.Networks.mutation.Mutable_Network>`): the object to be saved

        Return:
            None
        """
        def add(data,key,value):
            data[key] = data.get(key,[])+[value]

        with shelve.open(self.data_file) as data:
            add(data,'generation',t_gen)
            add(data,'fitness',net.fitness)
            add(data,'n_interactions',len(net.dict_types.get('Interaction',[])))
            add(data,'n_species',len(net.dict_types.get('Species',[])))

        with open(self.bests_file.format(t_gen),'wb') as freezer:
            pickle.dump(net,freezer)

    def save_restart_file(self, kgeneration, header, tgeneration):
        """Save a dbm file, keyed by the generation number (a string!) and with value a
        [parameter dictionary, genus].  Might be more transparent to write out Poulation instance and
        forget header, and be sure to update tgeneration
        """
        rprmt = dict(header = header,
                     state = random.getstate(),
                     tgeneration = tgeneration)
        with shelve.open(self.restart_file, writeback=True) as restart_data:
            restart_data[str(kgeneration)] = (rprmt, self.genus)

        print('restart file saved after generation=', kgeneration, 'next tgeneration=', tgeneration)

    def pop_sort(self):
        """Sort the population with respect to fitness"""
        self.genus.sort(key=lambda X: X.fitness if X.fitness is not None else 9999)

    def update_fitness(self,nnetwork,integration_result):
        """Update (in place) the fitness and the dlt_fitness

        Args:
            nnetwork (int): the index of the network in the population
            integration_result (list): the output of compile_and_integrate

        Returns:
            None: in place modification
        """
        if integration_result:
            current_fitness = float(integration_result[0])
            self.genus[nnetwork].data_evolution = [data for data in integration_result] #stores various data on evolution
        else: #catches the None fitness
            current_fitness = None
            self.genus[nnetwork].data_evolution = None

        #update dlt_fitness and handle the None fitness option
        if current_fitness and self.genus[nnetwork].fitness:
            self.genus[nnetwork].dlt_fitness = current_fitness - self.genus[nnetwork].fitness
        elif current_fitness:
            self.genus[nnetwork].dlt_fitness = 9999
        elif self.genus[nnetwork].fitness:
            self.genus[nnetwork].dlt_fitness = -9999
        else:
            self.genus[nnetwork].dlt_fitness = 0
        self.genus[nnetwork].fitness = current_fitness

    def genus_mutate_and_integrate(self,prmt,nnetwork,mutation=True):
        """mutate, and update the fitness of one individual

        Args:
            prmt (dict): the inits parameters for integration
            nnetwork (int): the index of the network in the population
            mutation (bool): a flag to activate mutation

        Returns:
            int: the number of mutation
            int: the index of the network in the population
            Network: The resulting network after mutation
        """
        [n_mutations,nnetwork,mutated_net,result]=self.genus[nnetwork].mutate_and_integrate(prmt,nnetwork,self.tgeneration,mutation)
        if n_mutations:
            mutated_net.flag_mutation = True
        self.update_fitness(nnetwork,result)
        self.n_mutations+=n_mutations
        return [n_mutations,nnetwork,mutated_net]

    def pop_mutate_and_integrate(self,initial,first_mutated,last_mutated,prmt,net_stat):
        """ Recompute the fitness for half the population and mutate/compute the fitness for the rest. Save all the data in net_stat

        Args:
            initial (int): index of the first individual in population
            first_mutated (int): index of the first mutated individual in population
            last_mutated (int): index of the last mutated individual in population
            prmt (dict): the inits parameters for integration
            net_stat (NetworkStat): to store the population data

        Returns:
            None: in place modification
        """
        self.n_mutations=0
        for nnetwork in range(initial,first_mutated):
                self.genus_mutate_and_integrate(prmt,nnetwork,mutation=False)
        for nnetwork in range(first_mutated,last_mutated):
            self.genus_mutate_and_integrate(prmt,nnetwork,mutation=True)
        for nnetwork in range(initial,last_mutated):
            net_stat.add_net(self.genus[nnetwork])
        return None

    def initialize_identifier(self):
        """
        Set an unique index to every network of the initial population an set the max_network_identifier
        value. If the run restarts an existing simulation, only max_network_identifier is computed.
        """
        if prmt["restart"]["activated"]:
            self.max_network_identifier = max([net.identifier for net in self.genus])
            for net in self.genus:
                net.flag_mutation = False
        else:
            for i,net in enumerate(self.genus):
                net.identifier = i
                net.flag_mutation = False
            self.max_network_identifier = len(self.genus)-1

    def increment_identifier(self,network):
        """
        Test whether the network was mutated. If so the network identifier
        is updated with a new index.
        """
        if network.flag_mutation:

            self.max_network_identifier+=1
            network.parent = network.identifier
            network.identifier = self.max_network_identifier
            network.flag_mutation = False


    def evolution(self,prmt):
        """
        Main method to evolve population

        Return:
            None
        """
        #added for selection
        with open("usr_options.json",'r') as infile:
            option = json.load(infile)
        #---

        first_mutated = int( self.npopulation * (1-prmt['frac_mutate']) )
        net_stat = pop_stat.NetworkStat(stat_dict)
        gen_stat = pop_stat.GenusStat()

        self.initialize_identifier()
        #initialize attributs for each network needed in loop over generations
        if self.same_seed:
            print('Best fitness prior to mutations=', self.genus[0].fitness)
        else:
            self.pop_mutate_and_integrate(0,self.npopulation,self.npopulation-1,prmt,net_stat)
            for nnetwork in range( self.npopulation ):
                self.genus[nnetwork].data_next_mutation=self.genus[nnetwork].compute_next_mutation()
            self.pop_sort()
            print('Best/worst fitness prior to mutation=', self.genus[0].fitness, self.genus[-1].fitness)
        # MAIN EVOLUTIONARY LOOP
        start_gen = max(self.generation0,prmt["restart"]["kgeneration"])
        prmt["restart"]["kgeneration"] = 0

        #added for selection
        archive_threshold = 1.2 
        #1.2 is good for the current slection methods but can be added as a hyperparameter to initialization or -s tag
        #---

        for t_gen in range(start_gen,prmt['ngeneration']):
            prmt['generation'] = t_gen
            net_stat = pop_stat.NetworkStat(stat_dict)
            gen_stat = pop_stat.GenusStat()
            for net in self.genus:
                setattr(net,"gen",t_gen)
            # mutate a fraction of networks in population (those least fit)
            if (prmt['redo']==1): # in this case, recompute the fitness of non-mutated ind.
                self.pop_mutate_and_integrate(0,first_mutated,self.npopulation,prmt,net_stat)
            else: #only mutation
                self.pop_mutate_and_integrate(first_mutated,first_mutated,self.npopulation,prmt,net_stat)
            print("Total number of mutations in the population :%i"%self.n_mutations)

            # Adjust the tgeneration time to have roughly one mutation per individual in pop
            if (self.n_mutations>0):
                self.tgeneration=self.tgeneration*self.npopulation*prmt['frac_mutate']/self.n_mutations
            else:
                self.tgeneration=2*self.tgeneration
            fitness_treatment(self)
            
            #added for selection
            #hyperparameters are to be updated here as opposed to inizialization c file
            if 'novelty_search' not in option.keys():
                print('not using novelty sorting fitness for opt')
                archive_threshold = self.pop_sort(tgen, archive_threshold)
           
            elif option['novelty_search'] == 'fitness_novelty':
                print('using novelty sorting fitness')
                archive_threshold = self.fitness_novelty(t_gen,archive_threshold, pareto=False, k=100, logit=True)
            
            elif option['novelty_search'] == 'fitness_novelty_pareto':
                print('using novelty sorting fitness with pareto')
                archive_threshold = self.fitness_novelty(t_gen,archive_threshold, pareto=True, k=100, logit=False)
            
            elif option['novelty_search'] == 'ts_autoencoder':
                print('using novelty: selecting most novel timeseries using autoencoder')
                archive_threshold = self.timeseries_embed(t_gen,archive_threshold, pareto=True,k=100,linear=False)

            elif option['novelty_search'] == 'ts_dba':
                print('using novelty: selecting most novel timeseires with DTW Barycenter averaging')
                archive_threshold = self.dba_novelty(t_gen,archive_threshold,k=5)

            elif option['novelty_search'] == 'ts_dtw':
                print('using novelty: selecting most novel timeseires with DTW')
                archive_threshold = self.dtw_novelty(t_gen,archive_threshold,custom=False,k=100,pareto=True)
           
            elif option['novelty_search'] == 'graph_ged':
                print('using novelty: selecting moset novel graph with GED')
                archive_threshold = self.ged_novelty(t_gen,archive_threshold)
           
            elif option['novelty_search'] == 'random_selection':
                print('selecting individuals at random with blacklist for nones')
                self.random_sort()

            elif option['novelty_search'] == 'ts_arima':
                print('using novelty: selecting most novel timeseries using SARIMAX model')
                archive_threshold = self.arima_embed(t_gen,archive_threshold, pareto=True, k=100)
           
            else:
                print("You didn't input any of the possible choices will default to fitness selection, if the issue persists contact tech support")
                archive_threshold = self.pop_sort(t_gen, archive_threshold)

            
            self.getadd_best()
            self.save_bests()
            #---


            gen_stat.process_sorted_genus(self)

            # print info after mutation step so built_integrator*.c consistent with Bests file

            seed = int(re.search("Seed(\d+)",prmt["workplace_dir"]).group(1)) # extract seed from worplace_dir name
            test_STOP_file(prmt["stop_file"],dict(seed=seed,generation=t_gen,fitness=self.genus[0].fitness))
            header = "\nAfter generation {0:d} Best fitness={1}".format(t_gen,self.genus[0].fitness)
            if __verbose__:
                header+="data=[]\n"
                for data in self.genus[0].data_evolution:
                    if data and len(data) > 0:
                        header=header+"data.append("+data+")\n"
            print(header)
            print("New generation time: %f"%self.tgeneration)
            sys.stdout.flush()
            self.storing(t_gen,self.genus[0])

            # Handling of different options
            try:
                if prmt['pareto'] and prmt['freq_plot']:
                    if t_gen % prmt['freq_plot'] == 0:
                        self.pop_print_pareto(self.namefolder+'/pareto'+str(t_gen),self.namefolder+'/rank1_nets'+str(t_gen))
            except KeyError:
                pass
            # Selection step, replace less fit networks by the fitter ones.
            for nnetwork in range( self.npopulation//2 ):
                self.increment_identifier(self.genus[nnetwork])
                self.genus[-1-nnetwork]=copy.deepcopy(self.genus[nnetwork]) # duplicates best half

            for individual in self.genus:
                new_seed = int(random.random()*100000) #generates new seed  to be sure not to overlap
                individual.Random=random.Random(new_seed) #reinitializes the random generator of every network

            # print statistics for this generation.  Need wrap all these variables into generic stat.
            if(t_gen%prmt['freq_stat'] == 0):
                net_stat.output()
                print("Total number of mutations: %i"%self.n_mutations)
                gen_stat.output()

            # save an exact copy of genus and relevant parameters for continuing loop
            if( t_gen%prmt['restart']['freq'] == 0):
                self.save_restart_file( t_gen, header, self.tgeneration)
            sys.stdout.flush()
