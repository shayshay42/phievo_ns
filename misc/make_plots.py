import matplotlib.pyplot as plt
import ast
import os
import pickle

# path_to_base = '/Users/owner1/Documents/GitHub/phievo/'
# workfile = 'example_lac_operon_fullyrandomcontrol/'
# seed = 'Seed3/'
# workspace = path_to_base+workfile+seed
# outputs = 'outputs_forplotting'
# pathtofile = path_to_base+workfile+seed+outputs

def plot_bests(workspace):
	fitness = []
	sparseness = []
	for i in range(220):
		with open(workspace+'/Bests_'+str(i)+'.net', 'rb') as bf: net = pickle.load(bf)
		fitness.append(net.fitness)
		sparseness.append(net.sparseness)

	gen = list(range(1,len(fitness)+1))

	plt.plot(gen, fitness, '-o',linewidth = 1, markersize=1)
	plt.ylabel('fitness')
	plt.title("Most fit network's fitness per generation while selecting for most sparse")
	plt.xlabel('generation')
	plt.savefig(workspace+'/best_fitness.png')
	#plt.show()
	#for sparsest
	plt.clf()

	plt.plot(gen, sparseness, '-o',linewidth = 1, markersize=1) #weird that num gen diff
	plt.ylabel('sparseness')
	plt.title("Most fit network's sparseness per generation while selecting for most sparse")
	plt.xlabel('generation')
	plt.savefig(workspace+'/best_sparseness.png')
	#plt.show()
	plt.clf()

def plot_latent(workspace,pathtofile):
	# for file.pkl in population_directory:
	# 	get dataevolution 250long timeseries and the latent outputs
	# 	perform PCA or UMAP or tSNE or ICA
	# 	plot PCs
	# 	check with plot of timeseries

	# can also just check sparseness and PCA correlation? no 
	pass

def plot_train_hist():
	workspace = "/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/GRU_pareto_archivethreshold_40/Seed0/"
	mins = []
	maxs = []
	for i in range(173):
		with open(workspace+'/'+str(i)+'train_hist.pkl', 'rb') as hf: hist = pickle.load(hf)
		mins.append(min(hist['train']))
		maxs.append(max(hist['train']))
	plt.plot(list(range(173)),mins)
	plt.plot(list(range(173)),maxs)
	plt.savefig(workspace+'/train_hist.png')
	plt.show()
	pass

def make_frames_from_net(workspace,pathtofile):
	with open(os.path.join(workspace, 'fitests.pkl'), 'rb') as df: fitests = pickle.load(df)
	with open(os.path.join(workspace, 'sparsests.pkl'), 'rb') as df: spests = pickle.load(df)
	with open(os.path.join(workspace, 'finalstatearchive.pkl'), 'rb') as df: arch = pickle.load(df)
	most_fits_fitness = []
	most_fits_sparseness = []
	most_sparses_fitness = []
	most_sparses_sparseness = []
	for nf,ns in zip(fitests,spests):
		most_fits_fitness.append(nf.fitness)
		most_fits_sparseness.append(nf.sparseness)
		most_sparses_fitness.append(ns.fitness)
		most_sparses_sparseness.append(ns.sparseness)
	
	arch_fitness = []
	arch_sparseness = []
	for na in arch:
		arch_fitness.append(na.fitness)
		arch_sparseness.append(na.sparseness)
	return most_fits_fitness,most_fits_sparseness,most_sparses_fitness,most_sparses_sparseness,arch_fitness,arch_sparseness

def get_plots_from_frame(workspace,selection):
	ff,fs,sf,ss,af,asp = make_frames_from_net(workspace,selection)
	gen = list(range(1,len(ff)+1))

	plt.plot(gen, ff, '-o',linewidth = 1, markersize=1)
	plt.ylabel('fitness')
	plt.title("Most fit network's fitness per gen while selecting"+selection)
	plt.xlabel('generation')
	plt.savefig(workspace+'/fitest_fitness.png')
	#plt.show()
	#for sparsest
	plt.clf()

	plt.plot(gen, fs, '-o',linewidth = 1, markersize=1) #weird that num gen diff
	plt.ylabel('sparseness')
	plt.title("Most fit network's sparseness per gen while selecting"+selection)
	plt.xlabel('generation')
	plt.savefig(workspace+'/fitest_sparseness.png')
	#plt.show()
	plt.clf()
	#for just sparsest's fitmess

	tot = sum(list(range(len(ff))))
	colors = [i/tot for i in gen]
	#print(colors)
	plt.scatter(ff, fs, c=gen,s=1, cmap='viridis')#linewidth = 0.5
	plt.ylabel('sparseness')
	plt.title('Scatter of (fitness, sparseness) for highest fitness agents per gen'+selection)
	plt.xlabel('fitness')
	plt.savefig(workspace+'/fitest_fitsparse.png')
	#plt.show()
	plt.clf()

	plt.plot(gen, sf, '-o',linewidth = 1, markersize=1)
	plt.ylabel('fitness')
	plt.title("Most sparse network's fitness per gen while selecting"+selection)
	plt.xlabel('generation')
	plt.savefig(workspace+'/sparsest_fitness.png')
	#plt.show()
	#for sparsest
	plt.clf()

	plt.plot(gen, ss, '-o',linewidth = 1, markersize=1) #weird that num gen diff
	plt.ylabel('sparseness')
	plt.title("Most sparse network's sparseness per gen while selecting"+selection)
	plt.xlabel('generation')
	plt.savefig(workspace+'/sparsest_sparseness.png')
	#plt.show()
	plt.clf()
	#for just sparsest's fitmess

	tot = sum(list(range(len(sf))))
	colors = [i/tot for i in gen]
	#print(colors)
	plt.scatter(sf, ss, c=gen,s=1, cmap='viridis')#linewidth = 0.5
	plt.ylabel('sparseness')
	plt.title('Scatter of (fitness, sparseness) for highest sparseness agents per gen'+selection)
	plt.xlabel('fitness')
	plt.savefig(workspace+'/sparsest_fitsparse.png')
	#plt.show()
	plt.clf()

	binses = 100
	plt.hist(asp, bins=binses)
	tit = 'count for '+str(binses)+' bins'
	plt.ylabel(tit)
	plt.title('Archive Sparseness')
	plt.xlabel('Sparseness of the archived agents above sparseness threshold(dynamic)'+selection)
	plt.savefig(workspace+'/sparseness_archive_bin=100.png')
	#plt.show()
	plt.clf()

	binses = 100
	plt.hist(af, bins=binses)
	tit = 'count for '+str(binses)+' bins'
	plt.ylabel(tit)
	plt.title('Archive Fitness')
	plt.xlabel('Fitness of the archived agents above sparseness threshold(dynamic)'+selection)
	plt.savefig(workspace+'/fitness_archive_bin=100.png')
	#plt.show()
	plt.clf()

def save_plots(workspace,pathtofile):
	outs = open(pathtofile, 'r')
	lines = outs.readlines()
	#index depends on what was copy pasted
	# archive = [float(i) for i in lines[2][2:-2].split(', ')]
	fitness = [float(i) for i in lines[4][2:-2].split(', ')]
	sparse = [float(i) for i in lines[6][2:-2].split(', ')]
	fit_sparse = [float(i) for i in lines[8][2:-2].split(', ')]

	gen = list(range(1,len(fitness)+1))


	#for fitesst while selecting with sparseness

	plt.plot(gen, fitness, '-o',linewidth = 1, markersize=1)
	plt.ylabel('fitness')
	plt.title('Best fitness per generation while selecting for most sparse')
	plt.xlabel('generation')
	plt.savefig(workspace+'/best fitness per generation with sparse selection.png')
	#plt.show()
	#for sparsest
	plt.clf()

	plt.plot(list(range(1,len(sparse)+1)), sparse, '-o',linewidth = 1, markersize=1) #weird that num gen diff
	plt.ylabel('sparseness')
	plt.title('highest sparseness value per generation')
	plt.xlabel('generation')
	plt.savefig(workspace+'/best sparseness per generation with sparse selection.png')
	#plt.show()
	plt.clf()
	#for just sparsest's fitmess

	plt.plot(list(range(1,len(fit_sparse)+1)), fit_sparse, '-o',linewidth = 1, markersize=1)
	plt.ylabel('fitness of sparsest')
	plt.title('Fitness of the sparsest agent per generation')
	plt.xlabel('generation')
	plt.savefig(workspace+'/best sparses fitness per generation with sparse selection.png')
	#plt.show()
	plt.clf()
	#for archive

	# binses = 100
	# plt.hist(archive, bins=binses)
	# tit = 'count for '+str(binses)+' bins'
	# plt.ylabel(tit)
	# plt.title('Archive Space')
	# plt.xlabel('fitness of the sparse agents above threshold(dynamic)')
	# plt.savefig(workspace+'/archive_bin=100.png')
	# #plt.show()
	# plt.clf()
	#for sparse - fit
	tot = sum(list(range(len(sparse))))
	colors = [i/tot for i in list(range(1,len(fit_sparse)+1))]
	#print(colors)
	plt.scatter(sparse, fit_sparse, c=list(range(1,len(fit_sparse)+1)),s=1, cmap='viridis')#linewidth = 0.5
	plt.ylabel('fitness')
	plt.title('Scatter of (sparseness, fitness) for highest sparseness agents per generation')
	plt.xlabel('sparseness')
	plt.savefig(workspace+'/sparse_fitness.png')
	#plt.show()
	plt.clf()

# a = "C:/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/fitness_novelty_paretok10_lacoperon_archivethreshold_1point2/Seed0/output_forplotting.txt"
# b = "C:/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/fitness_novelty_paretok10_lacoperon_archivethreshold_1point2/Seed0"
# save_plots(b,a)


#"/mnt/c/Users/boop/Sync/default/PauLF_lab/working_NS_fe/phievo/example_lac_operon_lowerdtw/Seed0/output_forplotting.txt"