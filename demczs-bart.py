import numpy as np
import numpy.random as npr
import multiprocessing as mp
import time as tm
import ctypes
import sys, os
import imp
import random
##################### BART SPECIFIC ##############################
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir + "/../modules/MCcubed/src")
from BARTfunc2 import BARTclass
##################################################################
class my_process(mp.Process):
    def __init__(self,itterations,thinning,data,ind,errors,function,\
                 chi_func,con_func,const,ex,smstp,queuet,z_array,shape,\
                 M,num_chain,num_split,pre_hist,chi_array):
        mp.Process.__init__(self)
        #global z_array
        self.its      = itterations
        self.thinning = thinning
        self.data     = data
        self.ind      = ind
        self.errors   = errors
        ############## BART specific, set the extra argument as the conf
        ############## file. comment out the function assignment, as it is
        ############## set later in the bart init
	#self.ex       =	ex
        self.conffile = ex[0]
        #self.function = function
        self.chi_func = chi_func
        self.con_func = con_func
        self.const    = const
        self.ex       = ex
        self.smstp    = smstp
        self.queuet   = queuet
        self.lock     = M.get_lock()
        self.M        = M
        self.nc       = num_chain
        self.num_sp   = num_split
        self.pre_hist = pre_hist
        self.z_array  = z_array
        self.shape    = shape
        self.z_view   = np.ctypeslib.as_array(z_array.get_obj())
        self.zview    = self.z_view.reshape(shape)
        self.chi_arr  = np.ctypeslib.as_array(chi_array.get_obj())
        self.zlocs    = []
        for i in range(self.num_sp):
            self.zlocs.append([])
        
    def run(self):
	#These tasks must be done at the beginning of the run function
        #as it is not until the run function starts that the program
        #actually forks, thus if we need tasks to happen in a new process
        #they must be done here
        #The numpy random system must have its seen relinitialized in the
        #new process or each of the sub processes will inherit the same
        #seed, and produce the same 'random' steps. Such is the downfall
        #of random numbers in computers. The random.randomint function is
        #process and thread safe and will produce different random numbers
        #in each thread.
	self.myname = mp.current_process().name.split("-")[-1]
	np.random.seed(random.randint(0,100000))
        self.z_view   = np.ctypeslib.as_array(self.z_array.get_obj())
        self.zview    = self.z_view.reshape(self.shape)

        ########## BART specific for initializing needed objects
        self.bartc = BARTclass(self.conffile)
        self.function = self.bartc.run
        self.bartc.my_init()

        self.par_p    = []
        self.chi_p    = []

        for i in xrange(self.num_sp):
            randomint      = np.random.randint(self.M.value)
            self.par_p.append(np.copy(self.zview[randomint]))
            model          = self.function(self.par_p[i],self.ind,self.ex)
            self.chi_p.append(self.chi_func(model,self.data,self.errors,self.ex))
            self.chi_p[i] += self.con_func(self.par_p[i],self.const)
        self.par_c    = np.copy(self.par_p)
        self.chi_c    = np.copy(self.chi_p)
        self.chi_b    = np.copy(self.chi_p)
        self.par_b    = np.copy(self.par_p)
        self.gamma    = 2.38/np.sqrt(2*len(self.par_p[0]))
        self.accept   = [0]*self.num_sp
        self.gen      = 0
        stop_con      = self.its/self.thinning
        npru = np.random.uniform
        jrange = xrange(self.num_sp)
        irange = xrange(self.thinning)
        con_func = self.con_func
        function = self.function
        chi_func = self.chi_func
        snooker  = self.snooker
        gamma    = self.gamma
        nc       = self.nc
        getvec   = self.getvec
        while self.gen < stop_con:
            start_time= tm.time()
            for j in jrange:
                for i in irange:
                    if npru() < 0.1:
                        self.par_c[j] = self.par_p[j] + snooker(j)
                    else:
                        if npru() < 0.2:
                            bamma = 1
                        else:
                            bamma = gamma*1/(nc*2)
                        self.par_c[j] = self.par_p[j] + getvec(bamma)
		    con_check = con_func(self.par_c[j],self.const)
		    if con_check == 0:
			try:
                    		model = function(self.par_c[j],self.ind,self.ex)
		    		self.chi_c[j]  = chi_func(model,self.data,
                                               self.errors,self.ex)
                                del model
			except:
				print("model failed failed",self.par_c[j])
				self.chi_c[j] = 9e99
		    else:
			self.chi_c[j]  = con_check
                    #self.chi_c[j] += self.con_func(self.par_c[j],self.const)
                    alpha = np.exp(-0.5*(self.chi_c[j] - self.chi_p[j]))
                    if not np.isfinite(alpha):
                        alpha = 0.0
                    if alpha >=1 or alpha > npru():
                        self.accept[j] += 1
                        self.chi_p[j] = self.chi_c[j]
                        self.par_p[j][:] = self.par_c[j][:]
                        if self.chi_c[j]  < self.chi_b[j]:
                            self.chi_b[j] = self.chi_c[j]
                            self.par_b[j][:] = self.par_c[j][:]
                    del alpha
            self.update()
        self.cleanup()
        return
        
    def cleanup(self):
        self.queuet.send([self.par_b,self.chi_b,self.accept,self.zlocs])
        ##################### BART specific to free up memory
        self.bartc.trm.free_memory()

            
    def update(self):
        with self.lock:
            for l in xrange(self.num_sp):
                self.zview[self.M.value][:]    = self.par_p[l][:]
                self.chi_arr[self.M.value-self.pre_hist]  = self.chi_p[l] 
                self.zlocs[l].append(self.M.value)
                self.M.value += 1
        self.gen += 1
                    
    def snooker(self,j):
        randint = np.random.randint(self.M.value-self.nc)
        ints    = self.getrandints(0,self.M.value)
        #ints    = np.random.permutation(self.M.value)
	direc   = self.par_p[j] - self.zview[randint]
	while np.sum(direc) == 0.0:
		randint = np.random.randint(self.M.value-self.nc)
		direc = self.par_p[j] - self.zview[randint]
        zp1     = np.dot(self.zview[ints[0]],direc)/\
                  np.dot(direc,direc)*\
                  direc
        zp2     = np.dot(self.zview[ints[1]],direc)/\
                  np.dot(direc,direc)*\
                  direc
        return np.random.uniform(1.2,2.2)*(zp1-zp2)

    def getrandints(self,low,high):
        ints = np.random.randint(low,high,2)
        while ints[1] is ints[0]:
            ints[1] = np.random.randint(low,high)
        return ints

    def getvec(self,gamma):
        #ints = np.random.permutation(np.arange(0,self.M.value))
        ints = self.getrandints(0,self.M.value)
        retval = gamma*(self.zview[ints[0]][:] - self.zview[ints[1]][:]) 
        error = np.zeros((self.smstp.shape))
        for i in xrange(len(self.smstp)):
          if (self.smstp[i] == 0):
            error[i] = 0
          else:
            error[i] = np.random.normal(0,self.smstp[i])
        return retval + error


def demczs(iterations,data,ind,errors,function,chi_func,con_func,
         par,steps,const,ex,num_chain,thinning,num_processes=False,hist_mult=10):
    '''
    This is a more advanced function than the standard mcmc fitting routine.
    It should be fairly generic to use for multiple functions, just specify
    the modeling function, and the chi_sqare function. Implimentation of
    "Differential Evolution Markov Chain with snooker updater and fewer chains"
    by Cajo et. al 2008 DOI 10.1007/s11222-008-9104-9. There is one small change
    that I have made, where the gamma parameter is scalled by
    1/(3*num_of_parameters).I find that this speeds up the convergence rate.

    Inputs:
    itterations - (int)      number of steps in the markov chain
    data        - (array)    measured dependant variable 'observations'
    ind         - (array)    measured indipendant variables, the 'times' at
                             which to evaluate the function
    errors      - (array)    errors on the dependant measurements, if none set to
                             a ones array
    function    - (function) a python callable which should either be a function.
                             This is the function to fit arguments should be
                             (parameters, ind, extra) see note below.
    chi_func    - (function) A python function which can be called to generate chi
                             value (model,data,errors,extra) to the data. See note
                             below. Return value should be a float
    con_func    - (function) A python callable which will be used to constrain
                             the parameters, penalize the chi square if a
                             parameter goes outside the bounds. Callable
                             syntax should be con_func(par,bounds) where
                             par will be the list of parameters, bounds should
                             be a list of lists giving the bounds on each
                             parameter. See note below. Return value should be a float
    par         - (array)    list of initial guesses to go into the
                             function, used to get size of parameters and beginning
                             location. Should be length of number of variables.
    steps       - (array)    The sigma 'or stepsize' for each of the variables
                             in the function, controls the convergance rate of
                             the chain
    const       - (list)     List of lists giving the bounds on each parameter
                             used in conjunction with con_func
    ex          - (list)     This contains any extra values that may need to be
                             passed to the model func and chi func if you have no
                             extra values simply pass an empty list []
    num_chain   - (int)      The number of separate chains to run, this helps
                             the mcmc walk perpendicular to correlations and
                             also converge much faster. For demczs use a low number
                             of chains on the order of three
    thinning    - (int)      How many steps to run in each chain before updating the
                             global history. This helps save on memory, and changes
                             how the past itterations affect new steps, as the past
                             history is what generates the proposed steps at each step.
                             A low number will slow down efficiency as most of the time
                             will be spent communicating between processes.it also may
                             slow down convergence as the history will be too scattered
                             and not properly thinned. A large number
                             which is a large fraction of the number of itterations may
                             not provide enough history and will slow down convergence.
    num_proccess - (int)     If left False the number of seperate worker process will
                             be equal to the number of chains. If a number is specified
                             the number of chains will be split up over the number of
                             processes. If less chains than processes are specified,
                             the number of running chains will be set to the number of
                             processes.
    hist_mult   - (int)      This specifies how much starting history to create before
                             running the chains. The created history will be the
                             size no. of parameters times hist_mult, and will consist
                             of parameters drawn from uniform distributions on the
                             intervals defined in const

    Returns:
    par_b       - (array)    the best fit parameters for the function
    chi_b       - (float)    the best chi square value associated with par_b
    accept_rate - (float)    the percentage of steps in the chain that were
                             accepted, shoot for 50%, control with sigmas from
                             step
    par_arr     - (array)    All the saved parameters from mcmc loop at each
                             itteration, useful for determining errors
    chi_values  - (array)    array of the chi square value at each itteration
                             useful to see the trending behaivor of the mcmc
                             This array will be size no. of parameters times
                             hist_mult smaller than par_b, as the chi values
                             are only generated in the active chains and not
                             the pre generated history
    chain_locs  - (list)     This is a list of lists containing the indexes
                             in the parameter array (first dimension), and
                             chi array which correspond to each chain. I.E.
                             chain_locs[0] will be a list like
                             [50,53,57,...] which give the indicies where
                             the enteries in chain 0 can be found. This
                             array is useful if you want to reconstruct
                             the behaivor of each chain.


    ################ Note ################
    We have attempted to provide as much flexability in this function as possible.
    We are letting you specify the model and chi square generators such that you
    may handel complex data structures in the data, ind, and error arrays. Only
    the functions you specify see what is contained in these variables, and as
    such you are free to create complex structures such as lists of unequal size
    arrays, or some form of a class object. As such it is completely your
    responsability to construct these functions to work with your data set. If
    your functions need extra variables or arguments, put them in the list ex.
    It will be your responsibility to remember what order to parse the ex variable.

    Along with the flexability to use whatever structure you want, we give you
    the ability to specify the constraint function. This may be a prior you have
    on your data or a simple boundary function. This constraint function should
    return a value which is to be added to the chi square for a given set of
    parameters. A simple boundary case may be the function loops over all the
    parameters given to it, and checks if they are within the boundaries supplied.
    For each parameter outside the boundary add a large value to the chi penalty
    such that that step will be rejected. This is known as an uninformative prior.
    A more refined case may be penalizing a parameter by an increasing amount based
    on how far away it is from a given value you specify in your function, or as
    calculated from your constraints. This is known as an informative prior.

    ################ Note2 ################
    The python multiprocessing module is nice, and fast, and distributed with python
    but has a few short commings. On windows there exists no true fork command
    as there is in unix. As such all functions you want to pass to demczs must be
    imported from a file on the disk, such that they can be reimported by the child
    processes. This is especially an issue when using demczs in conjunction with
    ipython. Dynamically defined functions (if you copy and paste them into the ipython
    session) will not work. Likewise lambda functions, dynamically typed or imported
    will not work, as they are not re-importable. Demczs itself must be imported from
    a function and not dynamically typed as well, for the same reasons above. Working
    in ipython is not out of the question however! If you use the ipython magic commands
    %load_ext autoreload, then %autoreload 2, ipython will reimport an edited module
    (file) each time a change is made and saved. This still allows for the flexability
    to dynamically investigate data, while using the benifits of multiprocessing.
    The one exception to this is if you do something to break or hang the child processes,
    such as returning an array of values from the supplied chi_function instead of
    a value. Thie child processes will crash, and the main process will hang. In the
    future I may put in checks for this so that it will be handeled better, but for
    now if you have the inputs correct, and return floats from chi_func and con_func,
    this should not be an issue. If a crash does occur, ipython must be restarted. In
    the future I may also add sanitation checks to the inputs and outputs as well. For
    now just be mindful.At this time, I only know of this bug on windows, but similar
    caviats apply about multiprocessing on other platforms.
    '''
    time_start = tm.time()
    #global z_array
    if num_processes == False:
        num_processes = num_chain
    if num_chain < num_processes:
        num_processes = num_chain
    if num_chain > num_processes:
        split = []
        div   = int(np.floor(num_chain/num_processes))
        if np.mod(num_chain,num_processes):
            split += [div]*(num_processes-1)
            split += [div+1]
        else:
            split += [div]*num_processes
    steps = np.array(steps)
    if num_chain == num_processes:
        split = np.ones(num_chain,dtype=int)
    iterations = np.floor(iterations/thinning)*thinning
    z_array = mp.Array(ctypes.c_double,np.zeros((len(par)*hist_mult+\
                    num_chain*iterations/thinning)*\
                       len(par)))
    chi_arr = \
    mp.Array(ctypes.c_double,np.zeros(num_chain*iterations/thinning))

    #z_view  = np.frombuffer(z_array,dtype='float64')
    z_view  = np.ctypeslib.as_array(z_array.get_obj())
    zview   = z_view.reshape((len(par)*hist_mult+num_chain*iterations/thinning,\
                              len(par)))
    #chi_arr = np.zeros(num_chain*iterations/thinning)
    #populate the initial zarray
    M = mp.Value('i',hist_mult*len(par))
    for i in range(hist_mult*len(par)):
        for j in range(len(par)):
            zview[i][j] = npr.uniform(const[j][0],const[j][1])
    #make the queues
    qtree_there  = [mp.Pipe() for x in range(num_processes)]
    pipe_end = {"master":1,"worker":0}
    myproc = []
    for i in range(num_processes):
        myproc.append(my_process(iterations,\
                                 thinning,\
                                 data,\
                                 ind,\
                                 errors,\
                                 function,\
                                 chi_func,\
                                 con_func,\
                                 const,\
                                 ex,\
                                 steps/10.,\
                                 qtree_there[i][pipe_end["worker"]],\
                                 z_array,\
                                 zview.shape,\
                                 M,\
                                 num_chain,\
                                 split[i],\
                                 hist_mult*len(par),\
                                 chi_arr))
        tm.sleep(0.1)
        myproc[-1].start()
    gen_counter = 0
    target = 0.05
    start_time = tm.time()
    itovrthn = iterations/thinning
    '''
    while gen_counter < itovrthn:
        if gen_counter/(itovrthn) >target:
            now = tm.time()
            print(str(round(100*target))+'% '+\
                  str(round((now-start_time)/60./(target)*(1-target),2))\
                  +'m remaining')
            target += 0.05
        for n in range(num_processes):
            tmp_value = qtree_there[n][pipe_end["master"]].recv()
            for l in range(len(tmp_value[0])):
            #    print(M,np.sum(split[:n]),l)
                zview[M+np.sum(split[:n])+l][:] = tmp_value[0][l][:]
                chi_arr[M+np.sum(split[:n])+l-hist_mult*len(par)]  = tmp_value[1][l] 
        M += num_chain
        for n in range(num_processes):
            qtree_back[n][pipe_end["master"]].send(M)
        gen_counter += 1
    '''
    stop_condition = len(par)*hist_mult+num_chain*iterations/thinning
    while M.value < stop_condition:
        #sleep here until all the processes are done with their work
        if M.value/float(stop_condition) > target:
            print('{} % complete'.format(target*100))
            target += 0.05
        tm.sleep(0.1)
    #now we must get the results
    best_list   = []
    chi_list    = []
    accept_list = []
    chain_locs  = []
    for n in range(num_processes):
        tmp_value = qtree_there[n][pipe_end["master"]].recv()
        for l in range(len(tmp_value[0])):
            best_list.append(tmp_value[0][l])
            chi_list.append(tmp_value[1][l])
            accept_list.append(tmp_value[2][l])
            chain_locs.append(tmp_value[3][l])
        p = myproc[n]
        p.join()
    min_pos = np.argmin(chi_list)
    return best_list[min_pos],chi_list[min_pos],\
      np.sum(accept_list)/(num_chain*iterations),np.copy(zview),np.copy(chi_arr),chain_locs




if __name__ == '__main__':            
    output_test =  demczs(1e3,dep,ind,depe,pole_model,pole_chi,pole_prior,\
                  par,step,con,ex,3,25)
