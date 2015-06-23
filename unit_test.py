#because of a limitation of multiprocessing in python on windows
#demczs must be in a seperatly imported file as well as the functions
#that are passed to it, so we import them here
from demczs import *
import unit_test_functions as utf
import numpy as np
import numpy.random as npr
import time as tm

#construct a dataset
#independant variable
size  = 1000
indp  = np.linspace(-4,4,size)
#set the known parameters
par_k = [7,1,15]


#create the dependant variable, [] is the extra unused argumnet
#function is defined in unit_test_functions.py
dep   = utf.fit_func(par_k,indp,[])

#create some noise we know well
var   = 4
noise = npr.normal(0,var,size)
#this is going to be or vector of error guesses
sigma = np.ones(size)*var

#now put the signal together with the nosie
dep_n = dep + noise

#here are some guesses for the parameter function as well as bounds on them
#and guess on step sizes for each parameter and an extra empty array since
#none of our functions require extra arguments
par_g  = [1.8,np.pi/2.7,5.2]
bounds = [[6.5,7.5],[0.5,1.5],[14,16]]
steps  = [0.01,0.01,0.01]
extra  = []
#number of chains
num_chain = 4
#set the thinning rate must be lower than number of itterations if uncertan use 1-10% of
#the number of itterations
thinning = 100


#now we want to output the parameters with using number of processes equal to number of chians
t1 = tm.time()
output = demczs(7e4,dep_n,indp,sigma,utf.fit_func,utf.chi_func,utf.con_func,par_g,steps,\
                bounds,extra,num_chain,thinning)
t2 = tm.time()

#now we want to output the parameters with using number of processes not equal to number of chians
t3 = tm.time()
output2 = demczs(3e4,dep_n,indp,sigma,utf.fit_func,utf.chi_func,utf.con_func,par_g,steps,\
                 bounds,extra,num_chain,thinning,num_processes=2)
t4 = tm.time()

print("Results of one process per chain\n")
print("Time elapsed:"+str(t2-t1)+"s\n")
print("True parameters "+str(par_k[0])+","+str(par_k[1])+","+\
      str(par_k[2])+"\n")
print("Estimated Parameters +"+str(np.median(output[3][-500:,0]))+","+\
      str(np.median(output[3][-500:,1]))+","+str(np.median(output[3][-500:,0]))+"\n")
print("Estimated uncertanties +"+str(np.std(output[3][-500:,0]))+","+\
      str(np.std(output[3][-500:,1]))+","+str(np.std(output[3][-500:,0]))+"\n\n")
print("Results of multiple chain per process\n")
print("Time elapsed:"+str(t4-t3)+"s\n")
print("True parameters "+str(par_k[0])+","+str(par_k[1])+","+\
      str(par_k[2])+"\n")
print("Estimated Parameters +"+str(np.median(output2[3][-500:,0]))+","+\
      str(np.median(output2[3][-500:,1]))+","+str(np.median(output2[3][-500:,0]))+"\n")
print("Estimated uncertanties +"+str(np.std(output2[3][-500:,0]))+","+\
      str(np.std(output2[3][-500:,1]))+","+str(np.std(output2[3][-500:,0]))+"\n")
