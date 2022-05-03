from phievo import __silent__,__verbose__
if __verbose__:
    print("Execute classes_eds2.py")
from phievo.initialization_code import display_error
from importlib import import_module
import phievo.networkx as nx
import numpy as np
import string,copy,sys
import pickle
import os

import phievo.Networks.PlotGraph as PlotGraph
from phievo.initialization_code import *
from phievo.Networks.classes_eds2 import *
from phievo.AnalysisTools import palette
from phievo.Networks.interaction import *

import shelve
import sys,os,glob,pickle,zipfile,re
from urllib.request import urlretrieve
from phievo.AnalysisTools import palette
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from  matplotlib.lines import Line2D

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
import random
import phievo.networkx as nx #.algorithms.similarity.graph_edit_distance as ged #not sure if proper
from phievo.AnalysisTools import Simulation
import matplotlib.pyplot as plt
from make_plots import save_plots
from networkx import graph_edit_distance as ged

def read_network(filename,verbose=False):
    """Retrieve a whole network from a pickle object named filename
    Args:
        filename (str): the directory where the object is saved
    Returns:
        The stored network
    """
    with open(filename,'rb') as my_file:
        net = pickle.load(my_file)
    if verbose:
        print("Network retrieve from: {}".format(filename))
    return net


# path = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/results/results/lac_operon_novafterpopsort_1000gen_bestsofarforplots/Seed0/Bests_600.net"
# exp = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/results/results/lac_operon_novafterpopsort_1000gen_bestsofarforplots"

# path2 = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/lac_operon_4seeds_1pretty/Seed1/Bests_866.net"
# exp2 = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/lac_operon_4seeds_1pretty"

# netw = read_network(path2)
# netw.draw()

# path3 = '/Users/shay/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_0.net'
# exp3 = '/Users/shay/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon'

# netw = read_network(path3)

# sim = Simulation(exp3)

# dt = sim.run_dynamics(net=netw)
# time = dt['time']
# zz = dt[0][0][:,[dt['outputs'][0]]]#,dt['inputs'][0],dt['inputs'][1]]]
# plt.plot(time, zz, '-o',linewidth = 1, markersize=1)
# plt.ylabel('output 0 run_dynamics')
# plt.title('time series of output 0 from simulation')
# plt.xlabel('time')
# plt.savefig('timeseries_test_march18_comparing this is 250.png')
# plt.show()
# plt.clf()

# test_net1 = "C://Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_0.net"
# test_net2 = "C://Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_610.net"

# for i in range(10):
#     path0 = '/Users/shay/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_'+str(i)+'.net'
#     net1 = read_network(path0)
#     print(i)
#     net1.draw()

workspace = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/for_gm/fitness_novelty/Seed0/generation900/population/"
for i in range(100):
	net = read_network(workspace+'network'+str(i)+'.pkl')
	net.draw().savefig(workspace+'netw500'+str(i))

	ts = [net.data_evolution[i] for i in range(0,len(net.data_evolution),20)]
	plt.plot(list(range(len(ts))), ts)
	plt.show()

# path0 = '/Users/shay/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_0.net'
# path9 = '/Users/shay/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon/Seed0/Bests_4.net'

# net1 = read_network(path0)
# net1.draw()
# net2 = read_network(path9)
# net2.draw()
# print(type(net1))
# print(type(net1.graph))
# # print(net1.graph.__dict__)
# print(ged(net1.graph,net2.graph))

# zz = netw.data_evolution

# #print(zz)
# print(netw.fitness)

# plt.plot(list(range(len(zz))), zz, '-o',linewidth = 1, markersize=1)
# plt.ylabel('output 0 data_evolution')
# plt.title('time series of output 0 from simulation')
# plt.xlabel('time')
# plt.savefig('timeseries_test_march18_comparing this is 5000 found with direct retrieval.png')
# plt.show()

