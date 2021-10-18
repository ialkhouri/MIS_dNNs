
import networkx as nx
import numpy as np
import itertools
import time
import tensorflow as tf
from tensorflow.keras import layers
##################################################################################################
############################## steps: ############################################################
##################################################################################################

print('break')



##################################################################################################
##################################################################################################
############################## v9: no generator; solve like BOSS BUT with trainable weights and ones as inputs
############################### ONLY LP REDUCTION - here we add the simulated annealing
##################################################################################################

##################################################################################################
##################################################################################################
############################## Solution 2-Improvement local serch algorithm ###############################
##################################################################################################
##################################################################################################
def solution_improve_alg(X_star_thresholded, G, length):
    """

    :param MIS: the MIS solution list
    :param G: graph
    :return: increased size of the MIS if exists
    """

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    ### inputs: Graph, MIS
    MIS = list(np.argwhere(X_star_thresholded == 1).reshape(length, ))
    A = nx.to_numpy_array(G)
    list_of_replacement = []
    improv2_cntr = 0
    for MIS_node in MIS:
        # get combinations of nieghbors
        Neighbors = list(G[MIS_node])
        paris_of_Nieghbours = list(itertools.combinations(Neighbors, 2))
        for pair in paris_of_Nieghbours:
            # if they are not nighbours
            if A[pair[0], pair[
                1]] == 0:  # and len(intersection(MIS,[pair[0]])) == 0 and len(intersection(MIS,[pair[1]])) == 0:
                # get nighbors of node_j and node_l
                Neighbors_pair_0 = list(G[pair[0]])
                Neighbors_pair_1 = list(G[pair[1]])
                # tight-1 check on node_j and node_l
                if len(intersection(MIS, Neighbors_pair_0)) == 1 and len(intersection(MIS, Neighbors_pair_1)) == 1:
                    #improv2_cntr = improv2_cntr + 1
                    # remove node
                    list_of_replacement.append([MIS_node, pair[0], pair[1]])
                    #print("A replacement is taking place... node {} is removed and nodes {} and {} are added".format(
                    #    MIS_node, pair[0], pair[1]))
                    break
    ### here we need to loop over the potential list of replacements, remove and add, then check if we still have a MIS to update accordingly

    for potential_impr in list_of_replacement:
        # remove
        MIS_temp = MIS
        MIS_temp.remove(potential_impr[0])
        MIS_temp.append(potential_impr[1])
        MIS_temp.append(potential_impr[2])
        if MAXIMAL_IS_checker_2(MIS_temp,G) == 1:
            improv2_cntr = improv2_cntr + 1
            MIS = MIS_temp
            print("A replacement is taking place without harm ... node {} is removed and nodes {} and {} are added".format(
                potential_impr[0], potential_impr[1], potential_impr[2]))


    return improv2_cntr

def solution_improve_alg_2(MIS, G):
    """

    :param MIS: the MIS solution list
    :param G: graph
    :return: increased size of the MIS if exists
    """

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    ### inputs: Graph, MIS
    # MIS = list(np.argwhere(X_star_thresholded == 1).reshape(length, ))
    A = nx.to_numpy_array(G)
    list_of_replacement = []
    improv2_cntr = 0
    for MIS_node in MIS:
        # get combinations of nieghbors
        Neighbors = list(G[MIS_node])
        paris_of_Nieghbours = list(itertools.combinations(Neighbors, 2))
        for pair in paris_of_Nieghbours:
            # if they are not nighbours
            if A[pair[0], pair[
                1]] == 0:  # and len(intersection(MIS,[pair[0]])) == 0 and len(intersection(MIS,[pair[1]])) == 0:
                # get nighbors of node_j and node_l
                Neighbors_pair_0 = list(G[pair[0]])
                Neighbors_pair_1 = list(G[pair[1]])
                # tight-1 check on node_j and node_l
                if len(intersection(MIS, Neighbors_pair_0)) == 1 and len(intersection(MIS, Neighbors_pair_1)) == 1:
                    improv2_cntr = improv2_cntr + 1
                    # remove node
                    list_of_replacement.append([MIS_node, pair[0], pair[1]])
                    print("A replacement is taking place... node {} is removed and nodes {} and {} are added".format(
                        MIS_node, pair[0], pair[1]))
                    break

    return improv2_cntr

def solution_improve_alg_2_adv(MIS, G):
    """

    :param MIS: the MIS solution list
    :param G: graph
    :return: increased size of the MIS if exists
    """

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    ### inputs: Graph, MIS
    # MIS = list(np.argwhere(X_star_thresholded == 1).reshape(length, ))
    A = nx.to_numpy_array(G)
    list_of_replacement = []
    improv2_cntr = 0
    for MIS_node in MIS:
        # get combinations of nieghbors
        Neighbors = list(G[MIS_node])
        paris_of_Nieghbours = list(itertools.combinations(Neighbors, 2))
        for pair in paris_of_Nieghbours:
            # if they are not nighbours
            if A[pair[0], pair[
                1]] == 0:  # and len(intersection(MIS,[pair[0]])) == 0 and len(intersection(MIS,[pair[1]])) == 0:
                # get nighbors of node_j and node_l
                Neighbors_pair_0 = list(G[pair[0]])
                Neighbors_pair_1 = list(G[pair[1]])
                # tight-1 check on node_j and node_l
                if len(intersection(MIS, Neighbors_pair_0)) == 1 and len(intersection(MIS, Neighbors_pair_1)) == 1:
                    #improv2_cntr = improv2_cntr + 1
                    # remove node
                    list_of_replacement.append([MIS_node, pair[0], pair[1]])
                    print("A replacement is taking place... node {} is removed and nodes {} and {} are added".format(
                        MIS_node, pair[0], pair[1]))
                    break

    ### here we need to loop over the potential list of replacements, remove and add, then check if we still have a MIS to update accordingly

    for potential_impr in list_of_replacement:
        # remove
        MIS_temp = MIS
        MIS_temp.remove(potential_impr[0])
        MIS_temp.append(potential_impr[1])
        MIS_temp.append(potential_impr[2])
        #print("MIS before ppotential {} improvment = ".format(potential_impr), MIS)

        if MAXIMAL_IS_checker_2(MIS_temp,G) == 1:
            improv2_cntr = improv2_cntr + 1
            MIS = MIS_temp
            print("A replacement is taking place without harm ... node {} is removed and nodes {} and {} are added".format(
                potential_impr[0], potential_impr[1], potential_impr[2]), "MIS now = ", MIS)


    return improv2_cntr

##################################################################################################
##################################################################################################
##################################################################################################

##################################################################################################
##################################################################################################
############################## IS and Maximal IS checker fucntions ###############################
##################################################################################################
##################################################################################################
def IS_checker(X_star, G, th):
    X_star_thresholded = np.zeros(shape=(n_inputss))
    for i in range(n_inputss):
        if X_star[i] > th:
            X_star_thresholded[i] = 1

    list_of_nonZeros_nodes = list(np.nonzero(X_star_thresholded)[0])
    IS_Tester = 1

    # combinations to be checked from the X_star
    combos_list = list(itertools.combinations(list_of_nonZeros_nodes, 2))

    for pair in combos_list:
        if pair in G.edges:
            IS_Tester = 0
            #print("THIS IS NOT AN IS DUE TO THE EDGE BETWEEN", pair)
            break

    return IS_Tester

def MAXIMAL_IS_checker(X_star, G, th):
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
    X_star_thresholded = np.zeros(shape=(n_inputss))
    for i in range(n_inputss):
        if X_star[i] > th:
            X_star_thresholded[i] = 1



    list_of_nonZeros_nodes = list(np.nonzero(X_star_thresholded)[0])
    MAXIMAL_IS_Tester = 0
    flag = 0
    # check whether the nodes in list_of_nonZeros_nodes are actually an IS ? ; If not, exit the function with MAXIMAL_IS_checker == 0
    for node in list_of_nonZeros_nodes:

        #list_of_nonZeros_nodes_otherThanNode = np.setdiff1d(node, list_of_nonZeros_nodes)

        niebors = list(G[node])

        if intersection( list(G[node]) ,list_of_nonZeros_nodes) != []:
            #print("THIS IS NOT AN IS ")
            flag = 1
            break



    #MAXIMAL_IS_Tester = 0

    # below is to take out the case of having only one node !!!
    if len(list_of_nonZeros_nodes) > 1 and flag == 0:
        #### let the obtained Maximal-IS set be Q* with |Q|=L where L<n
        #### get Q^*

        nodes_Q_star = list_of_nonZeros_nodes
        nodes_G_not_in_Q_star = np.setdiff1d((G.nodes), nodes_Q_star)
        list_to_check = []
        for node in nodes_G_not_in_Q_star:
            # get the niebors
            niebors = list(G[node])
            if intersection(niebors, nodes_Q_star) != []:
                list_to_check.append(1)

        #print("WE are GOOD if number of ones = ", np.sum(list_to_check), " ; it should be = ", len(nodes_G_not_in_Q_star))

        if np.sum(list_to_check) == len(nodes_G_not_in_Q_star):
            MAXIMAL_IS_Tester = 1
            #print("we are good")

    return MAXIMAL_IS_Tester

def MAXIMAL_IS_checker_2(SET, G):
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    list_of_nonZeros_nodes = SET
    MAXIMAL_IS_Tester = 0
    flag = 0
    # check whether the nodes in list_of_nonZeros_nodes are actually an IS ? ; If not, exit the function with MAXIMAL_IS_checker == 0
    for node in list_of_nonZeros_nodes:

        # list_of_nonZeros_nodes_otherThanNode = np.setdiff1d(node, list_of_nonZeros_nodes)

        niebors = list(G[node])

        if intersection(list(G[node]), list_of_nonZeros_nodes) != []:
            # print("THIS IS NOT AN IS ")
            flag = 1
            break

    # MAXIMAL_IS_Tester = 0

    # below is to take out the case of having only one node !!!
    if len(list_of_nonZeros_nodes) > 1 and flag == 0:
        #### let the obtained Maximal-IS set be Q* with |Q|=L where L<n
        #### get Q^*

        nodes_Q_star = list_of_nonZeros_nodes
        nodes_G_not_in_Q_star = np.setdiff1d((G.nodes), nodes_Q_star)
        list_to_check = []
        for node in nodes_G_not_in_Q_star:
            # get the niebors
            niebors = list(G[node])
            if intersection(niebors, nodes_Q_star) != []:
                list_to_check.append(1)

        # print("WE are GOOD if number of ones = ", np.sum(list_to_check), " ; it should be = ", len(nodes_G_not_in_Q_star))

        if np.sum(list_to_check) == len(nodes_G_not_in_Q_star):
            MAXIMAL_IS_Tester = 1
            # print("we are good")

    return MAXIMAL_IS_Tester



# # ################################### Load graph
G = nx.read_gpickle("/directory/graph.gpickle")

# # ################################## plotting the graph with
# nx.draw(G,with_labels=True)
# plt.draw()
# plt.show()

#################### get the complimiant of G for the third connections:

N_number_of_nodes = len((G.nodes))
M_number_of_edges = len((G.edges))

M_number_of_edges_comp = (N_number_of_nodes*(N_number_of_nodes-1)/2) - M_number_of_edges

print('break')


##################################################################################################
##################################################################################################
############################## Constructing p from graph G
##################################################################################################

n_inputss   = N_number_of_nodes
m_outputs   = M_number_of_edges
m_outputs_c = int(M_number_of_edges_comp)

NN_output = 1

##################################################################################################
##################################################################################################
##################################################################################################

print('break')
##################################################################################################
##################################################################################################
################### MODIFY WEIGHTS of NN_p BASED ON GRAPH
##################################################################################################
##################################################################################################
idependent_set_size = 100*N_number_of_nodes # THIS IS SET TO HAVE THE MAXIMAL !!!

idependent_set_size = N_number_of_nodes**2

#### This is the numpy array we need to update the weights of the first set
first_set_of_weights = np.zeros(shape=(n_inputss,n_inputss+m_outputs+m_outputs_c))

#### add here from graph:
for i in range(n_inputss):
    first_set_of_weights[i,i] = 1.0 # tis stays the same


#### adding the connection from nodes in the graph to the first bottom m relu functions (edges in G)
idx=0
for pair in G.edges:
    first_set_of_weights[pair[0], n_inputss + idx] = 1.0
    first_set_of_weights[pair[1], n_inputss + idx] = 1.0
    idx=idx+1

#### adding the connection from nodes in the c.graph to the second bottom relu functions (edges in G')
idx=0
for pair in list(itertools.combinations(list(range(n_inputss)), 2)):
    if pair not in G.edges:
        first_set_of_weights[pair[0], n_inputss + m_outputs + idx] = 1.0
        first_set_of_weights[pair[1], n_inputss + m_outputs + idx] = 1.0
        idx = idx + 1


#### This is the numpy array we need to update the biases of the first set
first_set_of_biases = np.zeros(shape=(n_inputss+m_outputs+m_outputs_c))
first_set_of_biases[0:n_inputss] = -0.5                             # this for nodes
first_set_of_biases[n_inputss:n_inputss+m_outputs+m_outputs_c] = -1.0 # this is for edges of G and edges of G'


#### This is the numpy array we need to update the weights of the second set
second_set_of_weights = np.zeros(shape=(n_inputss+m_outputs+m_outputs_c , 1))


#### add here from graph:
second_set_of_weights[:,0][0:n_inputss ]                      = -1.0
second_set_of_weights[:,0][n_inputss:  n_inputss+m_outputs]   = N_number_of_nodes
second_set_of_weights[:,0][n_inputss+m_outputs :          ]   = -1.0


second_set_of_biases = np.zeros(shape=(1))


W_1 = tf.convert_to_tensor(first_set_of_weights, dtype=tf.float32)
W_2 = tf.convert_to_tensor(second_set_of_weights, dtype=tf.float32)
b_1 = tf.convert_to_tensor(first_set_of_biases , dtype=tf.float32)
b_2 = tf.convert_to_tensor(second_set_of_biases , dtype=tf.float32)


def my_init_W_1(shape , dtype=tf.float32):
    return W_1
def my_init_W_2(shape , dtype=tf.float32):
    return W_2
def my_init_b_1(shape , dtype=tf.float32):
    return b_1
def my_init_b_2(shape , dtype=tf.float32):
    return b_2

##########################################################################################################
##########################################################################################################
############################################################## build the NN
##########################################################################################################
##########################################################################################################


Z_shape_input  = n_inputss+m_outputs+m_outputs_c


##################################################################3################################################
################################3 initialize NN ##################################################################3
##################################################################3################################################
#################### this is the initial weights values (trainable weights): select as random if high dense graphs
##################################################################3################################################

number_of_initilizations = 4

input_set_of_weights_vectors = []
Solution_pool_length         = []

# # ############## (1) ##### this random
# np.random.seed(seed=1)
# input_set_of_weights_vectors.append(np.random.uniform(low=0.9, high=1.0, size=(n_inputss,)))
# input_set_of_weights_vectors.append(np.random.uniform(low=0.91, high=1.0, size=(n_inputss,)))
# np.random.seed(seed=2)
# input_set_of_weights_vectors.append(np.random.uniform(low=0.9, high=1.0, size=(n_inputss,)))
# input_set_of_weights_vectors.append(np.random.uniform(low=0.95, high=1.0, size=(n_inputss,)))

###############  (2)  ##### initialize as w(n) =  1-(deg(n)/max_deg_of_graph), \forall n \in [N]
for seed in [10 ,11, 212, 11213]:
## this is a way to get the degree per node: len(list(G.degree._nodes[0].items()))
    list_of_degrees = []
    np.random.seed(seed=seed)
    for i in range(0, n_inputss):
        list_of_degrees.append(G.degree[i])
    max_deg = np.max(list_of_degrees)
    input_set_of_weights_vector = np.zeros(shape=(n_inputss,))
    for i in range(0, n_inputss):
        ## To prevent exact repititive probs: add some very small epsilon
        input_set_of_weights_vector[i] = 1 - (G.degree[i] / max_deg) + np.random.uniform(low=0.0, high=0.1)
        # input_set_of_weights_vector[i] = 1 - (G.degree[i] / max_deg)

    input_set_of_weights_vectors.append(input_set_of_weights_vector)
    #input_set_of_weights_vectors.append(1-input_set_of_weights_vector)


for init_index in range(number_of_initilizations):




    ############################################ this is needed always
    W_0_vec = tf.convert_to_tensor(input_set_of_weights_vectors[init_index], dtype=tf.float32)
    def my_init_W_0_vec(shape , dtype=tf.float32):
        return W_0_vec


    ##################################################################3################################################
    ################################3 initialize NN ##################################################################3
    ##################################################################3################################################

    combined_NN_p = tf.keras.Sequential()
    layer1 = layers.MyDenseLayer(num_outputs = n_inputss,
                                kernel_initializer=my_init_W_0_vec,
                                 kernel_constraint=tf.keras.constraints.zeroOne())
    layer1.trainable=True
    combined_NN_p.add(layer1)

    layer2 = layers.Dense(units=n_inputss+m_outputs+m_outputs_c ,
                   activation='relu',
                   name="connect1",
                   kernel_initializer=my_init_W_1,
                   use_bias=True,
                   bias_initializer=my_init_b_1)
    layer2.trainable=False
    combined_NN_p.add(layer2)

    layer3 = layers.Dense(units=1,
                   activation=None,
                   name="connect2",
                   kernel_initializer=my_init_W_2,
                   use_bias=True,
                   bias_initializer=my_init_b_2)
    layer3.trainable=False
    combined_NN_p.add(layer3)



    #### combine the NN

    ################# optimization
    # good and fast results on most is found at initial_learning_rate = 0.0010
    initial_learning_rate = 0.001
    opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    combined_NN_p.compile(optimizer=opt, loss="MSE")

    print("Break: compiling is done here for init = ", init_index)


    ##################################################################################################
    ##################################################################################################
    ##################################################################################################

    batch_size_gen = 1
    batch_size_2 = batch_size_gen

    ################################################################
    ##### X_train is the same for both gen and combined models #####
    ################################################################

    X_train = np.ones(shape=(1,n_inputss))

    ############################################################
    ### Y_train_combined is the v_d (desired value)
    ################################################################

    P_desired = np.zeros(shape=(1))
    P_desired[0] = -idependent_set_size/2


    Y_train_combined = np.zeros(shape=(batch_size_2,1,1))

    Y_val_combined = P_desired.reshape(1,1)
    for i in range(batch_size_2):
        Y_train_combined[i,:,:] = Y_val_combined

    Y_train_combined = Y_train_combined.reshape(batch_size_2,1)

    Y_desired = Y_val_combined
    Y_val_combined = Y_val_combined.reshape(1,1,1)
    Y_val_combined = Y_val_combined.reshape(1,1)
    #Y_val_combined = Y_val_combined.reshape(3)
    print('break: preparing repeated dataset is done for init', init_index)


    ################################################################
    ### train
    ################################################################


    training_steps = 100


    start = time.time()
    for i in range(training_steps):

        combined_NN_p.fit(X_train, Y_train_combined, epochs=2*n_inputss, batch_size=1, verbose=0)

        x = combined_NN_p.layers[0].get_weights()[0]


        BOSS_current_output = x
        v_theta             = combined_NN_p(np.ones(shape=(n_inputss)))


        X_star = BOSS_current_output.reshape(n_inputss)

        if any(X_star >1.0 ) or any(X_star < 0.0 ):
            print("exit AT training step = ", i, "; THERE IS A A VALUE IN THE WEIGHTS OUTSIDE [0,1]")
            break

        # check MAXIMAL-IS multiple thresholds !!!
        MAXIMAL_IS_tester = [MAXIMAL_IS_checker(X_star, G, 0.3),
                             MAXIMAL_IS_checker(X_star, G, 0.4),
                             MAXIMAL_IS_checker(X_star, G, 0.5),
                             MAXIMAL_IS_checker(X_star, G, 0.6),
                             MAXIMAL_IS_checker(X_star, G, 0.7)]

        if IS_checker(X_star, G, 0.5) == 1:
            X_star_thresholded_IS = np.zeros(shape=(n_inputss))
            for ii in range(n_inputss):
                if X_star[ii] > 0.5:
                    X_star_thresholded_IS[ii] = 1
            length_IS = np.count_nonzero(X_star_thresholded_IS)
            length_IS_w_Imp = np.count_nonzero(X_star_thresholded_IS) + solution_improve_alg(X_star_thresholded_IS, G,
                                 np.count_nonzero(X_star_thresholded_IS))
            print("IS is found at step ", [i], "with size = ", length_IS, 'total with improvement = ', length_IS_w_Imp)

        ## exit if we already have a Maximal-IS
        if any(np.array(MAXIMAL_IS_tester)==1):
            winning_threshold_temp = np.argmax(np.array(MAXIMAL_IS_tester))
            winning_threshold = [0.3, 0.4, 0.5, 0.6, 0.7][winning_threshold_temp]
            end = time.time()
            X_star_thresholded = np.zeros(shape=(n_inputss))
            #X_MIN_VALUE_TEST   = np.zeros(shape=(n_inputss))
            for ii in range(n_inputss):
                if X_star[ii] > winning_threshold:
                    X_star_thresholded[ii] = 1
            length = np.count_nonzero(X_star_thresholded) + solution_improve_alg(X_star_thresholded, G,
                                                                                 np.count_nonzero(X_star_thresholded))

            print("MAXIMAL-IS IS FOUND AT training step = ", i, "; Cardinality Ours = ", [length], "for init = ", init_index)
            #if length >= some_value_if_known:
            Solution_pool_length.append(length)
            break

        print('training_step = ', i,  "value = ",
              v_theta.numpy()[0], ' ; desired value set for optimiztion = ', [P_desired])



print('Solution_pool = ', [Solution_pool_length])


print('break')


