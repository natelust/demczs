import numpy as np
import numpy.random as npr
import multiprocessing as mp
import time as tm
import ctypes
import random

class my_process(mp.Process):
    def __init__(self,itterations,thinning,data,ind,errors,function,\
                 chi_func,con_func,const,ex,smstp,queuet,queueb,z_array,shape,\
                 M,num_chain,num_split):
        mp.Process.__init__(self)
        #global z_array
        self.its      = itterations
        self.thinning = thinning
        self.data     = data
        self.ind      = ind
        self.errors   = errors
        self.function = function
        self.chi_func = chi_func
        self.con_func = con_func
        self.const    = const
        self.ex       = ex
        self.smstp    = smstp
        self.queuet   = queuet
        self.queueb   = queueb
        self.M        = M
        self.nc       = num_chain
        self.num_sp   = num_split
        self.z_array  = z_array
        self.shape    = shape
        self.z_view   = np.ctypeslib.as_array(z_array.get_obj())
        self.zview    = self.z_view.reshape(shape)
        #need to initialize a starting position and such
        self.par_p    = []
        self.chi_p    = []
        for i in range(self.num_sp):
            randomint      = np.random.randint(M)
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

    def run(self):
        #Though this is also done in the init function we find we must
        #also do it once the process actually starts, otherwise on windows
        #the shared memory which contains the history cant be read at
        #subsiquent itterations.
        np.random.seed(random.randint(0,100000))
        self.z_view        = np.ctypeslib.as_array(self.z_array.get_obj())
        self.zview    = self.z_view.reshape(self.shape)
        while self.gen < self.its/self.thinning:
            for j in range(self.num_sp):
                for i in range(self.thinning):
                    jump = False
                    if np.random.uniform() < 0.1:
                        self.par_c[j] = self.par_p[j] + self.snooker()
                    #elif np.random.uniform() < 0.7*self.gen/(self.its/self.thinning):
                    #    self.par_c[j] = self.bigjump(j)
                    #    jump = True
                    else:
                        if np.random.uniform() < 0.1:
                            bamma = 1
                        else:
                            bamma = self.gamma*1/(self.nc*3)
                        self.par_c[j] = self.par_p[j] + self.getvec(bamma)
                    model = self.function(self.par_c[j],self.ind,self.ex)
                    self.chi_c[j]  = self.chi_func(model,self.data,\
                                               self.errors,self.ex)
                    self.chi_c[j] += self.con_func(self.par_c[j],self.const)
                    alpha = np.exp(-0.5*(self.chi_c[j] - self.chi_p[j]))
                    #if jump == True:
                    #    pass
                    if np.isfinite(alpha) != True:
                        alpha = 0.0
                    if alpha >=1 or alpha > np.random.uniform():
                        self.accept[j] += 1
                        self.chi_p[j] = self.chi_c[j]
                        self.par_p[j][:] = self.par_c[j][:]
                    if self.chi_c[j]  < self.chi_b[j]:
                        self.chi_b[j] = np.float(self.chi_c[j])
                        self.par_b[j][:] = self.par_b[j][:]
            self.update()
        self.cleanup()
        return
        
    def cleanup(self):
        self.queuet.put([self.par_b,self.chi_b,self.accept])
            
    def update(self):
        self.queuet.put([self.par_p,self.chi_p])
        self.M = self.queueb.get()
        self.gen += 1
                    
    def snooker(self):
        randint = np.random.randint(self.M-self.nc)
        ints    = np.random.permutation(self.M)
        zp1     = np.dot(self.zview[ints[0]],self.zview[randint])/\
                  np.dot(self.zview[randint],self.zview[randint])*\
                  self.zview[randint]
        zp2     = np.dot(self.zview[ints[1]],self.zview[randint])/\
                  np.dot(self.zview[randint],self.zview[randint])*\
                  self.zview[randint]
        return np.random.uniform(1.2,2.2)*(zp1-zp2)

    def getvec(self,gamma):
        #if self.gen == 0:
        #    mult = 0
        #else:
        #    mult = self.gen*self.nc
        #print(mult,self.M)
        ints = np.random.permutation(np.arange(0,self.M))
        return gamma*(self.zview[ints[0]][:] - self.zview[ints[1]][:]) +\
               np.random.normal(0,self.smstp[:])

    def bigjump(self,j):
        ints = np.random.randint(self.M/1.1,self.M)
        #print(self.zview[ints])
        return self.zview[ints][:]

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
    #z_view  = np.frombuffer(z_array,dtype='float64')
    z_view  = np.ctypeslib.as_array(z_array.get_obj())
    zview   = z_view.reshape((len(par)*hist_mult+num_chain*iterations/thinning,\
                              len(par)))
    chi_arr = np.zeros(num_chain*iterations/thinning)
    #populate the initial zarray
    M = hist_mult*len(par)
    for i in range(hist_mult*len(par)):
        for j in range(len(par)):
            zview[i][j] = npr.uniform(const[j][0],const[j][1])
    #make the queues
    qtree_there  = [mp.Queue() for x in range(num_processes)]
    qtree_back  = [mp.Queue() for x in range(num_processes)]
    myproc = []
    for i in range(num_processes):
        p = my_process(iterations,\
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
                                 qtree_there[i],\
                                 qtree_back[i],\
                                 z_array,\
                                 zview.shape,\
                                 M,\
                                 num_chain,\
                                 split[i])
        myproc.append(p)
        tm.sleep(0.1)
        p.start()
    gen_counter = 0
    target = 0.1
    start_time = tm.time()
    while gen_counter < iterations/thinning:
        if gen_counter/(iterations/thinning) >target:
            now = tm.time()
            print(str(round(100*target))+'% '+\
                  str(round((now-start_time)/60./(target)*(1-target)*2,2))\
                  +'m remaining')
            target += 0.1
        for n in range(num_processes):
            tmp_value = qtree_there[n].get()
            #print(n,len(tmp_value))
            for l in range(len(tmp_value[0])):
                #print(M,np.sum(split[:n]),l)
                zview[M+np.sum(split[:n])+l][:] = tmp_value[0][l][:]
                chi_arr[M+np.sum(split[:n])+l-hist_mult*len(par)]  = tmp_value[1][l] 
        M += num_chain
        for n in range(num_processes):
            qtree_back[n].put(M)
        gen_counter += 1
    #now we must get the results
    best_list   = []
    chi_list    = []
    accept_list = []
    for n in range(num_processes):
        tmp_value = qtree_there[n].get()
        for l in range(len(tmp_value[0])):
            best_list.append(tmp_value[0][l])
            chi_list.append(tmp_value[1][l])
            accept_list.append(tmp_value[2][l])
        p = myproc[n]
        p.join()

    return best_list[np.argmin(chi_list)],np.min(chi_list),\
      np.sum(accept_list)/(num_chain*iterations),np.copy(zview),chi_arr


if __name__ == '__main__':            
    output_test =  demczs(1e3,dep,ind,depe,pole_model,pole_chi,pole_prior,\
                  par,step,con,ex,3,25)
