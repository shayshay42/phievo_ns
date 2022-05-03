"""
Defines selection function and auxillirary function used by said functions
these functions are added as methods to Population class in evolution_gillespie.py in Population Types directory

"""
import os

import random
import numpy as np
import phievo.networkx as nx 

from networkx import graph_edit_distance as ged
from networkx import optimize_graph_edit_distance as ged
from phievo.AnalysisTools import Simulation
import matplotlib.pyplot as plt
from phievo import AE
import random
import pandas as pd
import sys
sys.path.insert(1,os.path.join(os.getcwd(),'misc'))
import make_plots
from tqdm import tqdm
import torch
# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.datasets import CachedDatasets
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from torch import optim
import torchvision
import torch.nn as nn
import math
from math import log,sqrt
import json

# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from _ucrdtw import ucrdtw
import pickle
from scipy.spatial import distance

import networkx as nx

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

class selection_methods(object):

    def __init__(self): pass

    def getadd_best(self):
        sparse_list = [net.sparseness if net.fitness != None and len(net.data_evolution) != 0 else -99999 for net in self.genus]
        sparsest_net = self.genus[sparse_list.index(max(sparse_list))]
        self.sparsests.append(sparsest_net)
        print("most sparse's fitness is ___", sparsest_net.fitness, '___ and its sparseness is ___', sparsest_net.sparseness)

        fit_list = [net.fitness if net.fitness != None and len(net.data_evolution) != 0 else 99999 for net in self.genus]
        fitest_net = self.genus[fit_list.index(min(fit_list))]
        self.fitests.append(fitest_net)
        print("most fit's fitness is ___", fitest_net.fitness, '___ and its sparseness is ___', fitest_net.sparseness)

        print('and the last 5 in the archvive have fitness and sparsenss as follows', [(net.fitness, net.sparseness) for net in self.archive[-5:]])

    def save_bests(self):
        base = self.namefolder
        #os.mkdir(os.path.join(base,'plot_data'))
        with open(os.path.join(base,'fitests.pkl'), 'wb') as df: pickle.dump(self.fitests,df)
        with open(os.path.join(base,'sparsests.pkl'), 'wb') as df: pickle.dump(self.sparsests,df)
        with open(os.path.join(base,'finalstatearchive.pkl'), 'wb') as df: pickle.dump(self.archive,df)

    def pareto_nsga(self, sign=[1,1]):
        """
        not using crowd distance, strictly domination ranking
        """
        ranking = {}

        for i1, net1 in enumerate(self.genus):
            if net1.fitness == None: 
                ranking[i1] = 100 #dominated by all
                continue
            dom_count = 0
            # nd_indicator_dim1 = True
            # dom_count_dim1 = 0
            # nd_indicator_dim2 = True
            # dom_count_dim2 = 0
            for i2, net2 in enumerate(self.genus):
                if net2.fitness == None: continue #default domination
                if i1 == i2: continue
                if net1.sparseness < net2.sparseness and net1.fitness > net2.fitness:
                    dom_count += 1

            #     if net1.sparseness < net2.sparseness:
            #         nd_indicator_dim1 = False
            #         dom_count_dim1 += 1

            #     if net1.fitness < net2.fitness:
            #         nd_indicator_dim2 = False
            #         dom_count_dim2 += 1
            ranking[i1] = dom_count
            # if nd_indicator:
            #     non_dominated.append(net1)

        sorted_ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))
        #print(sorted_ranking)
        self.genus = [self.genus[indx] for indx in sorted_ranking.keys()]

    def pop_sort_archive(self,tgen, archive_threshold, logit=False, k=100):
        """
        Sort the population with respect to fitness
        while holding an archive with dynamic threshold
        and saving 
        """
        self.genus.sort(key=lambda X: X.fitness if X.fitness is not None else 9999)
        idx = 0
        for net_i in self.genus:
            ifit = net_i.fitness
            if logit and ifit!=None:
                ifit = log(1+net_i.fitness)
            if ifit == None:
                #print('none fitness for network ',idx)
                net_i.sparseness = 0
                continue
            # if ifit < 1:
            #     try:
            #         ifit = abs(log(ifit))
            #     except ValueError:
            #         print("Network number ",idx," solved the problem with fitnesss ",ifit)
            #         os.exit(0)
            dist_pernet = {}
            idx2 = 0
            for idx_j,net_j in enumerate(self.genus):
                jfit = net_j.fitness
                if log and jfit!=None:
                    jfit = log(1+net_j.fitness)
                if jfit == None:
                    dist_pernet[idx_j] = 0
                    #print('none fitness for network ',idx2)
                    continue
                # if jfit < 1:
                #     jfit = abs(log(jfit))
                dist_pernet[idx_j] = abs(ifit - jfit)
                idx2 += 1
            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")
            
            dist_to_archive = []
            for archive_net in self.archive: #don't know where to hold archive to make sure it doesn't get set to init
                archive_fit = archive_net.fitness
                dist_to_archive.append(abs(ifit - archive_fit))
            if len(dist_to_archive) != 0:
                    dist_pernet_L = list(dist_pernet.values())
                    dist_pernet_L.extend(dist_to_archive)
                    sorted_distance = sorted(dist_pernet_L)
                    sparcity= sum(sorted_distance[:k])/k
                    #sparcity += sum(dist_to_archive)/len(dist_to_archive)# used to be k-nn with k=n
            else:
                sorted_distance = dict(sorted(dist_pernet.items(), key=lambda item: item[1]))
                sparcity= sum(list(sorted_distance.values())[:k])/k
            net_i.sparseness = sparcity
            #sparse_dict[idx] = sparcity
            idx += 1
        
        if len(self.archive_log) > 4 and sum(self.archive_log[-4:]) < 4: #make these into parameters
            archive_threshold *= 0.9 #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        elif len(self.archive_log) > 4 and sum(self.archive_log[-4:]) > 10:
            archive_threshold *= 1.2
            print('raised archive_threshold:', archive_threshold)
            print('this is the last 5 in archive: \n', self.archive[-5:])
        archive_add = 0
        for net in self.genus:
            # if tgen <= 1: #add all behaviour from first generation to archive
            #     self.archive.append(net.fitness)
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net)
        self.archive_log.append(archive_add)
        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
        return archive_threshold

    def random_sort(self):
        not_nones = []
        nones = []
        for net in self.genus:
            if net.fitness == None:
                nones.append(net)
            else:
                not_nones.append(net)
        self.genus = random.sample(not_nones, len(not_nones))+nones

    def dtw_novelty(self,tgen, archive_threshold, custom=True, k=10, pareto=True):

        for net_i in tqdm(self.genus):

            i_series = net_i.data_evolution
            if i_series == None:
                net_i.sparseness = 0
                continue
            i_series = [i_series[i] for i in range(0,len(i_series),20)]

            dist_pernet = {}

            for idx_j,net_j in enumerate(self.genus):
                j_series = net_j.data_evolution
                if j_series == None:
                    dist_pernet[idx_j] = 0
                    continue
                j_series = [j_series[i] for i in range(0,len(j_series),20)]
                if custom:
                    dist_pernet[idx_j] = dtw_distance(i_series,j_series)
                else:
                    diss = ucrdtw(i_series,j_series,0.05,False)[1]
                    dist_pernet[idx_j] = diss
                    #print('its using _ucedtw', diss)

            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")


            
            dist_to_archive = []
            if len(self.archive) > 500:
                subset_arch = random.sample(self.archive,500)
            else:
                subset_arch = self.archive
            for archive_net in subset_arch: #don't know where to hold archive to make sure it doesn't get set to init
                arch_series = [archive_net.data_evolution[i] for i in range(0,len(archive_net.data_evolution),20)]
                if custom:
                        dist_to_archive.append(dtw_distance(i_series,arch_series))
                else:
                    dist_to_archive.append(ucrdtw(i_series,arch_series, 0.05, False)[1])
            if len(dist_to_archive) != 0:
                    dist_pernet_L = list(dist_pernet.values())
                    dist_pernet_L.extend(dist_to_archive)
                    sorted_distance = sorted(dist_pernet_L)
                    sparcity= sum(sorted_distance[:k])/k
                    #sparcity += sum(dist_to_archive)/len(dist_to_archive)# used to be k-nn with k=n
            else:
                sorted_distance = dict(sorted(dist_pernet.items(), key=lambda item: item[1]))
                sparcity= sum(list(sorted_distance.values())[:k])/k
            
            net_i.sparseness = sparcity
            #sparse_dict[idx] = sparcity            

        if not pareto: 
            self.genus.sort(key= lambda X: X.sparseness, reverse=True) #novelty selection
        else:
            self.pareto_nsga()

        if len(self.archive_log) > 4 and sum(self.archive_log[-4:]) < 4: #make these into parameters
            archive_threshold *= 0.9 #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        elif len(self.archive_log) > 4 and sum(self.archive_log[-4:]) > 10:
            archive_threshold *= 1.2
            print('raised archive_threshold:', archive_threshold)
            print('this is the last 5 in archive: \n', self.archive[-5:])

        archive_add = 0
        for net in self.genus:
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net)
        self.archive_log.append(archive_add)

        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)

        return archive_threshold

    def arima_embed(self, tgen, archive_threshold, k=100, pareto=True, look_log=5, archive_addition_rate = [6,20], lower_archive = 0.9, increase_archive=1.2):

        for ind, net in tqdm(enumerate(self.genus)):
            if net.data_evolution == None or net.fitness == None:
                continue
            #sm.tsa.statespace.SARIMAX
            model = ARIMA(
                [net.data_evolution[i] for i in range(0,len(net.data_evolution),20)],
                order=(2,1,3),
                seasonal_order=(0,0,0,0),#(1,1,1,1),
                enforce_stationarity=False,
                enforce_invertibility=False)
            results = model.fit(method_kwargs={"warn_convergence": False})
            #print(results.summary())
            emb = results.arparams.tolist()
            emb.extend(results.maparams.tolist())
            net.latent_output = emb

        idx = 0
        for net_i in self.genus:
            if net_i.data_evolution == None or net_i.fitness == None:
                continue
            dist_pernet = {}
            idx2 = 0
            for idx_j,net_j in enumerate(self.genus):
                if net_j.data_evolution == None or net_j.fitness == None:
                    continue
                dist = distance.euclidean(net_i.latent_output,net_j.latent_output)
                dist_pernet[idx_j] = dist
                idx2 += 1

            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")
            

            dist_to_archive = []
            if len(self.archive) > 500:
                subset_arch = random.sample(self.archive,500)
            else:
                subset_arch = self.archive
            for archive_net in subset_arch: #don't know where to hold archive to make sure it doesn't get set to init
                dist_to_archive.append(distance.euclidean(net_i.latent_output,archive_net.latent_output))
            if len(dist_to_archive) != 0:
                    dist_pernet_L = list(dist_pernet.values())
                    dist_pernet_L.extend(dist_to_archive)
                    sorted_distance = sorted(dist_pernet_L)
                    sparcity= sum(sorted_distance[:k])/k
                    #sparcity += sum(dist_to_archive)/len(dist_to_archive)# used to be k-nn with k=n
            else:
                sorted_distance = dict(sorted(dist_pernet.items(), key=lambda item: item[1]))
                sparcity= sum(list(sorted_distance.values())[:k])/k

            net_i.sparseness = sparcity

        if len(self.archive_log) > look_log and sum(self.archive_log[-look_log:]) < archive_addition_rate[0]: #make these into parameters
            archive_threshold *= lower_archive #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        elif len(self.archive_log) > look_log and sum(self.archive_log[-look_log:]) > archive_addition_rate[1]:
            archive_threshold *= increase_archive
            print('raised archive_threshold:', archive_threshold)
            print('this is the last 5 in archive: \n', [k.sparseness for k in self.archive[-5:]])

        archive_add = 0
        for net in self.genus:
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net) #add the whole network object to archive
        self.archive_log.append(archive_add)

        if pareto: #otherwise it fkn gets to an all NONE state
            self.pareto_nsga()
        else:
            self.genus.sort(key= lambda X: X.sparseness, reverse=True)
        
        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)

        return archive_threshold

    def timeseries_embed(self, tgen, archive_threshold, pareto=True, look_log=5, archive_addition_rate = [6,20], lower_archive = 0.9, increase_archive=1.2, netwo=2,test=True, linear=False, k=100):
        #gotta import simulation and use read_network

        # need each net in self.genus to have a sequence of numbers for the output node 
        # keep these sequences for each network in the generation and embed all with encoder bottleneck decoder

        #slow inefficient solution: store all networks(using restart function) and use analyzetools to run dynamics
        #network.simulation.run_dynamics()

        #better option but hard
        #self.genus[netn].data_evolution
        #but modify c files to output the history? trackout?

        #maybe already in result as buffer or stored to file

        #try reading data.db

        # import json
        
        # sim = Simulation("example_lac_operon") #should make it take options from the command line
        # # print('is this the timeseries for netwrok 3 of this genereation:', sim.run_dynamics(net=self.genus[2]))

        # if test == True and netwo == None:
        #     seqs_to_embed = dict()
        #     for idx,nnetw in tqdm(enumerate(self.genus)):
        #         if nnetw.fitness == None:
        #             continue
        #         dt = sim.run_dynamics(net=nnetw)
        #         seqs_to_embed[idx] = list(dt[0][0][:,dt['outputs'][0]])

        #     print("starting other method")






        for idx, nnetww in enumerate(self.archive): #take the bad ones out BUT DONT KNOW HOW THEY GOT THERE
            if type(nnetww.data_evolution) == 'NoneType':
                self.archive.remove(nnetww)





        seqs_to_embed = dict()
        for idx, nnetw in tqdm(enumerate(self.genus)):
            if nnetw.fitness == None:
                continue
            seqs_to_embed[idx] = [nnetw.data_evolution[i] for i in range(0,len(nnetw.data_evolution),20)]
        split = idx+1
        print('JEEZ how many are we adding:', len(self.archive))
        if len(self.archive) != 0:
            for inda,neta in enumerate(self.archive):
                idx=split+inda
                # print(idx, type(neta.data_evolution))
                # print(len(neta.data_evolution))
                try:
                    seqs_to_embed[idx] = [neta.data_evolution[i] for i in range(0,len(neta.data_evolution),20)]
                except TypeError:
                    continue
        # print(seqs_to_embed)
        
        # with open('seqs_to_embed_analpack.txt', 'w') as seqs_with_anal_file:
        #      seqs_with_anal_file.write(json.dumps(seqs_to_embed))

        # print("this is the second")

        # print(seqs_to_embed2)

        # with open('seqs_to_embed_subprocess.txt', 'w') as seqs_with_anal_2:
        #      seqs_with_anal_2.write(json.dumps(seqs_to_embed2))

        # os._exit(1)


        ste = pd.DataFrame.from_dict(seqs_to_embed)
        random.seed(208)
        val_ste = pd.DataFrame.from_dict({key:seqs_to_embed[key] for key in random.sample(list(seqs_to_embed.keys()),20)})
        #val_ste_sub = pd.DataFrame.from_dict(dict(random.sample(seqs_to_embed.items(), 20)))

        if not linear:
            #ste = np.array([v for v in seqs_to_embed.values()])
            #val_ste_sub = np.array(list(random.sample(seqs_to_embed.values(), 200)))
            train_dataset, seq_len, n_features = AE.create_dataset(ste)

            val_dataset, seq_len_val, n_features_val = AE.create_dataset(val_ste)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("building the model")
            #model = AE.RecurrentAutoencoder(seq_len, n_features, embedding_dim=128)
            model = AE.gru_simp()#lstm_simp()
            print("look at the model:", model)
            model = model.to(device)
            print("gonna start to train!")
            model_e, history = AE.train_recurAE(model,train_dataset,val_dataset,n_epochs=100)
            #print("is this the output?",model_e)
            base = self.namefolder
            with open(os.path.join(base,str(tgen)+'train_hist.pkl'),'wb') as df:
                pickle.dump(history, df)

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            train_dataset, seq_len, n_features = AE.create_dataset(ste)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

            model = AE.linearAE(input_shape=seq_len).to(device)
            #print("look at the model:", model)

            trained, stuff = AE.train_linear(model, device, seq_len, train_loader, n_epochs=250)
            # print("is this the output?",trained)
            # print("is THIS ONE the output?",stuff)

        for ind,se in seqs_to_embed.items():
            out = model.encoder(torch.tensor(se).to(device))
            # print('size of output', len(out), out)
            out_vector, hidden_state = out[0], out[1].tolist()

            if ind>=split: #update archive latents based on new embedding
                ida = ind-split
                self.archive[ida].latent_output = hidden_state
            #print("this is the latent_vector", latent_vector)
            else:
                self.genus[ind].latent_output = hidden_state

        # print(trained(torch.tensor(seqs_to_embed[0])).to(device))
        # print(stuff(train_dataset[0]))

        length = len(self.genus)
        a2a_dist = np.zeros((length,length))
        a2a_dict = {}
        for i,net1 in enumerate(self.genus):
            if net1.data_evolution == None or net1.fitness == None:
                net1.sparseness=0
            dist_pernet = []
            for j,net2 in enumerate(self.genus):
                if net2.data_evolution == None or net2.fitness == None:
                    net2.sparseness=0
                #dist = (net1.latent_output - net2.latent_output).pow(2).sum().sqrt()
                dist = distance.euclidean(net1.latent_output,net2.latent_output)
                a2a_dist[i,j] = dist
                a2a_dict[i] = dist
                dist_pernet.append(dist)
            
            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")

            if len(self.archive) > 500:
                subset_arch = random.sample(self.archive,500)
            else:
                subset_arch = self.archive
            for archive_net in subset_arch:
                dist_pernet.append(distance.euclidean(net1.latent_output,archive_net.latent_output))
                    #(net1.latent_output - net2.latent_output).pow(2).sum().sqrt())
            sorted_distance = sorted(dist_pernet)
            sparcity= sum(sorted_distance[:k])/k
            net1.sparseness = sparcity

        

        if len(self.archive_log) > look_log and sum(self.archive_log[-look_log:]) < archive_addition_rate[0]: #make these into parameters
            archive_threshold *= lower_archive #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        elif len(self.archive_log) > look_log and sum(self.archive_log[-look_log:]) > archive_addition_rate[1]:
            archive_threshold *= increase_archive
            print('raised archive_threshold:', archive_threshold)
            print('this is the last 5 in archive: \n', [k.sparseness for k in self.archive[-5:]])

        archive_add = 0
        for net in self.genus:
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net) #add the whole network object to archive
        self.archive_log.append(archive_add)

        if pareto: #otherwise it fkn gets to an all NONE state
            self.pareto_nsga()
        else:
            self.genus.sort(key= lambda X: X.sparseness, reverse=True)



                # (dt['time'], dt[0][0]) #takes only the zeroth should check 
                                                        #which the output is and then use thata key
        # seqs_to_embed
        # for netwo in self.genus:
        #     dt = sim.run_dynamics(net=netwo)
        # # time = dt['time']
        #     zz = dt[0][0][:,dt['outputs'][0]]

        # plt.plot(time, zz, '-o',linewidth = 1, markersize=1)
        # plt.ylabel('output 0 integration_result')
        # plt.title('time series of output 0 from simulation')
        # plt.xlabel('time')
        # plt.savefig('timeseries_test2.png')

        # print('is this the timeseries for netwrok 3 of this genereation:', sim.run_dynamics(net=self.genus[2]))

        #imported AE.py
        #embed

        # sim = Simulation("example_lac_operon") #should make it take options from the command line
        # dt = sim.run_dynamics(net=self.genus[2])
        # print(dt.keys())
        # print(len(dt['time']))
        # print(dt[0].keys())
        # print(len(dt[0][0]))
        # print('is this the timeseries for netwrok 3 of this genereation:', dt)
        
        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)

        return archive_threshold

    def ged_novelty(self,tgen, archive_threshold,look_log=5, archive_addition_rate=[6,20], lower_archive = 0.9, increase_archive=1.2):
        #add the attributes of the nodes in the GED calculation
        #use the optimize GED fucntion to make it faster or actually do kmeans or knn

        #.algorithms.similarity.graph_edit_distance as ged
        length = len(self.genus)
        a2a_dist = np.zeros((length,length))
        a2a_dict = {}
        for i,net1 in tqdm(enumerate(self.genus)):
            dist_pernet = []
            for j,net2 in enumerate(self.genus):
                #dist = ged(net1.graph,net2.graph)
                gen_ca = [v for v in nx.optimize_graph_edit_distance(net1.graph,net2.graph)]
                # print(gen_ca)
                dist = gen_ca[min([2,len(gen_ca)-1])]
                a2a_dist[i,j] = dist
                a2a_dict[i] = dist
                dist_pernet.append(dist)
            
            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")

            for archive_net in self.archive:
                gen_aa = [v for v in nx.optimize_graph_edit_distance(net1.graph,archive_net.graph)]
                dist_pernet.append(gen_aa[min([2,len(gen_aa)-1])])
            sparcity = sum(dist_pernet)/len(dist_pernet)
            net1.sparseness = sparcity

        self.genus.sort(key= lambda X: X.sparseness, reverse=True)
        self.pareto_nsga()

        # print('test', self.archive_log)
        # print('test2', self.archive_log[-5:])

        # if len(self.archive_log) > look_log: #and 
        if sum(self.archive_log[-int(look_log):]) < archive_addition_rate[0]: #make these into parameters
            archive_threshold *= lower_archive #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        # elif len(self.archive_log) > look_log:# and 
        elif sum(self.archive_log[-int(look_log):]) > archive_addition_rate[1]:
            archive_threshold *= increase_archive
            print('raised archive_threshold:', archive_threshold)
            #print('this is the last 5 in archive: \n', self.archive[-5:])

        archive_add = 0
        for net in self.genus:
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net) #add the whole network object to archive
        self.archive_log.append(archive_add)

        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)

        return archive_threshold #self.archive

    def fitness_novelty(self,tgen,archive_threshold,k=10,logit=True, pareto=False):
        """
        net_i.distance is the list of abs(net_i.fitness - net_j.fitness)
        computes k-nn sparseness (1/k)*sum(k smallest distances)
        sort networks based on sparseness
        add fitnesses above threshold to archive

        returns the updated archived behaviours(fitness values) 
        and sorts networks in population(genus) based on sparseness 
        for later selection in evolution gillespie
        """
        # sparse_dict = {}
        idx = 0
        for net_i in self.genus:

            ifit = net_i.fitness
            if logit and ifit!=None:
                ifit = log(1+net_i.fitness)
            if ifit == None:
                #print('none fitness for network ',idx)
                net_i.sparseness = 0

                continue

            # if ifit < 1:
            #     try:
            #         ifit = abs(log(ifit))
            #     except ValueError:
            #         print("Network number ",idx," solved the problem with fitnesss ",ifit)
            #         os.exit(0)

            dist_pernet = {}
            idx2 = 0
            for idx_j,net_j in enumerate(self.genus):

                jfit = net_j.fitness
                if log and jfit!=None:
                    jfit = log(1+net_j.fitness)
                if jfit == None:
                    dist_pernet[idx_j] = 0
                    #print('none fitness for network ',idx2)
                    continue

                # if jfit < 1:
                #     jfit = abs(log(jfit))

                dist_pernet[idx_j] = abs(ifit - jfit)
                idx2 += 1

            if len(dist_pernet) == 0:
                print('came up empty divide by zero error averted removed from population')
                dist_pernet = [0]
            
            if len(self.archive) == 0:
                print("archive is empty!")
            
            dist_to_archive = []
            for archive_net in self.archive: #don't know where to hold archive to make sure it doesn't get set to init
                archive_fit = archive_net.fitness
                dist_to_archive.append(abs(ifit - archive_fit))
            if len(dist_to_archive) != 0:
                    dist_pernet_L = list(dist_pernet.values())
                    dist_pernet_L.extend(dist_to_archive)
                    sorted_distance = sorted(dist_pernet_L)
                    sparcity= sum(sorted_distance[:k])/k
                    #sparcity += sum(dist_to_archive)/len(dist_to_archive)# used to be k-nn with k=n
            else:
                sorted_distance = dict(sorted(dist_pernet.items(), key=lambda item: item[1]))
                sparcity= sum(list(sorted_distance.values())[:k])/k

            net_i.sparseness = sparcity
            #sparse_dict[idx] = sparcity
            idx += 1


        #######SELECTION########

        if not pareto: 
            self.genus.sort(key= lambda X: X.sparseness, reverse=True) #novelty selection
        else:
            self.pareto_nsga()


        #_________________
        # rand_genus = []
        # nones= []
        # rand_order = [random.randint(0,len(self.genus)-1) for _ in range(len(self.genus))] 
        # for idx in rand_order: #random selection
        #     if self.genus[idx].fitness != None:
        #         rand_genus.append(self.genus[idx])
        #     else:
        #         nones.append(self.genus[idx])
        # self.genus = rand_genus+nones
        #_________________weighted random selection
        # gen_sparseness = [net.sparseness for net in self.genus]
        # tot_p = sum(gen_sparseness)
        # prob_dist = [sp/tot_p for sp in gen_sparseness]
        # self.genus = np.random.choice(self.genus, len(self.genus), p=prob_dist)



        #idk why random.shuffle doesnt work!

        #can save time by stopping the loop one if statement is false for the first tiem because genus os sprted
        if len(self.archive_log) > 4 and sum(self.archive_log[-4:]) < 4: #make these into parameters
            archive_threshold *= 0.9 #lowered by 10 percent
            print('lowered archive_threshold:', archive_threshold)
        elif len(self.archive_log) > 4 and sum(self.archive_log[-4:]) > 10:
            archive_threshold *= 1.2
            print('raised archive_threshold:', archive_threshold)
            print('this is the last 5 in archive: \n', self.archive[-5:])

        archive_add = 0
        for net in self.genus:
            # if tgen <= 1: #add all behaviour from first generation to archive
            #     self.archive.append(net.fitness)
            if net.sparseness >= archive_threshold:
                archive_add += 1
                self.archive.append(net)
        self.archive_log.append(archive_add)

        if tgen%100 == 0:
            base = self.namefolder
            os.mkdir(os.path.join(base,'generation'+str(tgen)))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'archive'))
            os.mkdir(os.path.join(base,'generation'+str(tgen),'population'))
            for i,net in enumerate(self.archive):
                with open(os.path.join(base,'generation'+str(tgen),'archive','archive_network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
            for i,net in enumerate(self.genus):
                with open(os.path.join(base,'generation'+str(tgen),'population','network'+str(i)+'.pkl'),'wb') as df:
                    pickle.dump(net, df)
        return archive_threshold #self.archive

    #drafts and not yet implemented
    def spectral():
        pass

    def dtw_distance(s, t):
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s[i-1] - t[j-1])
                # take last min from a square box
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[-1,-1]

    def svd_novelty(self):
        #no hyperparameter
        pass

    def dba_novelty(self):

        seqs_to_embed = dict()
        for idx, nnetw in tqdm(enumerate(self.genus)):
            if nnetw.fitness == None:
                continue
            seqs_to_embed[idx] = nnetw.data_evolution 

        seed = 0
        np.random.seed(seed)

        ste = pd.DataFrame.from_dict(seqs_to_embed)
        val_ste_sub = pd.DataFrame.from_dict(dict(random.sample(seqs_to_embed.items(), 20)))
        X_train = ste.astype(np.float32).to_numpy().transpose()
        sz = X_train.shape[1]

        #Pre-processing
        np.random.shuffle(X_train)
        # Keep only 50 time series
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        # Make time series shorter
        X_train = TimeSeriesResampler(sz=100).fit_transform(X_train)

        num_clusters = 3
        dtw_km = TimeSeriesKMeans(n_clusters=num_clusters,
                                  n_init=2,
                                  metric="dtw",
                                  verbose=True,
                                  max_iter_barycenter=10,
                                  random_state=seed)
        y_pred = dtw_km.fit_predict(X_train)

        for yi in range(num_clusters):
            plt.subplot(3, 3, 4 + yi)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)

            plt.plot(dtw_km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DTW $k$-means")

        print(dir(dtw_km))
        return y_pred


