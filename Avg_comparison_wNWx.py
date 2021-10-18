import keras.initializers
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import time
import dwave_networkx as dnx
from tensorflow.keras import Input, Model
from numpy import linalg as LA
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation,AveragePooling2D,Reshape
#from keras.layers import My
import cplex
from scipy.spatial.distance import jensenshannon
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy
##################################################################################################
############################## steps: ############################################################
##################################################################################################

#print('break')

# number of nodes to be tried
number_of_nodes_list = list(range(550,1050,25))
#number_of_nodes_list = [50, 100]

number_of_edges_list = 10*number_of_nodes_list

seed_list = list(range(0,10))

MIS_OSS_size = []
MIS_OSS_time = []
MIS_NWx_size = []
MIS_NWx_time = []

Avg_results_OSS_vs_NWx = []

for index in range(len(number_of_nodes_list)):
    for seed_value in seed_list:
        N_number_of_nodes_org = number_of_nodes_list[index]
        M_number_of_edges_org = number_of_edges_list[index]

        ################################### generate random graphs
        G = nx.generators.dense_gnm_random_graph(N_number_of_nodes_org, M_number_of_edges_org, seed=seed_value)

        # # ################################## plotting the graph with
        # nx.draw(G,with_labels=True)
        # plt.draw()
        # plt.show()



        N_number_of_nodes = len((G.nodes))
        M_number_of_edges = len((G.edges))

        M_number_of_edges_comp = (N_number_of_nodes*(N_number_of_nodes-1)/2) - M_number_of_edges



        #print('break')

        ##################################################################################################
        ##################################################################################################
        ############################## NetworkX solver for maximal-IS from 1992 paper:
        # ##############################Approximating maximum independent sets by excluding subgraphs
        ##############################
        ##################################################################################################
        #MaximalIS_nx = nx.maximal_independent_set(G)
        start_nx = time.time()
        Maximum_IS_nx = nx.approximation.maximum_independent_set(G)
        Maximum_IS_nx = list(Maximum_IS_nx)
        end_nx = time.time()
        #print("MAXIMAL [WITH TRIMMED] = ", MaximalIS_nx, 'with length = ', len(MaximalIS_nx)+pendant_nodes_cnt+isolated_cntr)
        #print("Maximum_IS_nx [WITH TRIMMED] = ", Maximum_IS_nx, 'with length = ', len(Maximum_IS_nx))
        #print("solution took = ", end_nx-start_nx, "seconds")
        #print("MAXIMAL [WITH TRIMMED] = ", MaximalIS_nx, 'with length = ', len(MaximalIS_nx)+pendant_nodes_cnt+NUMBER_of_nodes_tobe_added_from_LP_reduction)

        MIS_NWx_size.append(len(Maximum_IS_nx))
        MIS_NWx_time.append(end_nx-start_nx)



        ##################################################################################################
        ##################################################################################################
        ############################## Constructing p from graph G based on the theorem
        ##############################
        ##################################################################################################

        n_inputss   = N_number_of_nodes
        m_outputs   = M_number_of_edges
        m_outputs_c = int(M_number_of_edges_comp)

        ##################################################################################################
        ##################################################################################################
        ############################## IS and Maximal IS checker fucntions ###############################
        ##################################################################################################
        ##################################################################################################





        def IS_checker(X_star,G, th):
            X_star_thresholded = np.zeros(shape=(n_inputss))
            for i in range(n_inputss):
                if X_star[i] > th:
                    X_star_thresholded[i] = 1

            list_of_nonZeros_nodes = list(np.nonzero(X_star_thresholded)[0])
            IS_Tester = 1

            # combinations to be checked from the X_star
            combos_list = list(itertools.combinations(list_of_nonZeros_nodes,2))

            for pair in combos_list:
                if pair in G.edges:
                    IS_Tester = 0
                    print("THIS IS NOT AN IS DUE TO THE EDGE BETWEEN", pair)
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




        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        #print('break')
        ##################################################################################################
        ##################################################################################################
        ################### MODIFY WEIGHTS of NN_p BASED ON GRAPH
        ##################################################################################################
        ##################################################################################################
        idependent_set_size = N_number_of_nodes**2


        #### This is the numpy array we need to update the weights of the first set
        first_set_of_weights = np.zeros(shape=(n_inputss,n_inputss+m_outputs+m_outputs_c))

        ## add here from graph:
        # adding the connection from nodes in the graph
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
        #second_set_of_biases[0] = idependent_set_size/2
        #second_set_of_biases[0] = (n_inputss-1)/2

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

        ######## get the digonal matrix that represents the only trinable paramter in the combined net

        Z_shape_input  = n_inputss+m_outputs+m_outputs_c


        ##################################################################3################################################
        ################################3 initialize NN ##################################################################3
        ##################################################################3################################################
        #################### this is the initial weights values (trainable weights): select as random if high dense graphs
        ##################################################################3################################################

        ############## (1) ##### this random
        #input_set_of_weights_vector = np.random.uniform(low=0.0, high=1.0, size=(n_inputss,))


        ###############  (2)  ##### initialize as w(n) =  1-(deg(n)/max_deg_of_graph), \forall n \in [N]
        ## this is a way to get the degree per node: len(list(G.degree._nodes[0].items()))
        list_of_degrees=[]
        for i in range(0,n_inputss):
            list_of_degrees.append(G.degree[i])
        max_deg = np.max(list_of_degrees)
        input_set_of_weights_vector = np.zeros(shape=(n_inputss,))
        for i in range(0,n_inputss):
            ## To prevent exact repititive probs: add some very small epsilon
            input_set_of_weights_vector[i] = 1 - (G.degree[i]/max_deg) + np.random.uniform(low=0.0, high=0.05)
            #input_set_of_weights_vector[i] = 1 - (G.degree[i] / max_deg)


        ############################################ this is needed always
        W_0_vec = tf.convert_to_tensor(input_set_of_weights_vector, dtype=tf.float32)
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
        ####### optimization
        # good and fast results on most is found at initial_learning_rate = 0.0010
        initial_learning_rate = 0.0010
        opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        combined_NN_p.compile(optimizer=opt, loss="MSE")

        #print("break: compiling is done here")


        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

        batch_size_gen = 1
        batch_size_2 = batch_size_gen

        ################################################################
        ##### X_train is the vector of ones #####
        ################################################################

        X_train = np.ones(shape=(1,n_inputss))

        ############################################################
        ### Y_train is the h_d (desired value)
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
        #print('break: preparing repeated dataset is done')


        ############################################################
        ### train
        ################################################################


        training_steps = 5


        start = time.time()
        for i in range(training_steps):

            combined_NN_p.fit(X_train, Y_train_combined, epochs=2*n_inputss, batch_size=1, verbose=0)

            # the weights are to be checked here !!!!!!!! Be careful !!!!!!!!
            x = combined_NN_p.layers[0].get_weights()[0]

            BOSS_current_output = x
            v_theta             = combined_NN_p(np.ones(shape=(n_inputss)))

            X_star = BOSS_current_output.reshape(n_inputss)

            # check MAXIMAL-IS multiple thresholds !!!
            MAXIMAL_IS_tester = [MAXIMAL_IS_checker(X_star, G, 0.3),
                                 MAXIMAL_IS_checker(X_star, G, 0.4),
                                 MAXIMAL_IS_checker(X_star, G, 0.5),
                                 MAXIMAL_IS_checker(X_star, G, 0.6),
                                 MAXIMAL_IS_checker(X_star, G, 0.7)]

            ## exit if we already have a Maximal-IS
            if any(np.array(MAXIMAL_IS_tester)==1):
                winning_threshold_temp = np.argmax(np.array(MAXIMAL_IS_tester))
                winning_threshold = [0.3, 0.4, 0.5, 0.6, 0.7][winning_threshold_temp]
                end = time.time()
                X_star_thresholded = np.zeros(shape=(n_inputss))
                X_MIN_VALUE_TEST   = np.zeros(shape=(n_inputss))
                for ii in range(n_inputss):
                    if X_star[ii] > winning_threshold:
                        X_star_thresholded[ii] = 1
                        X_MIN_VALUE_TEST[ii] = 1/X_star[ii]
                length = np.count_nonzero(X_star_thresholded)

                MIS_OSS_size.append(length)
                MIS_OSS_time.append(end-start)

                Avg_results_OSS_vs_NWx.append(
                    [N_number_of_nodes, M_number_of_edges, seed_value, len(Maximum_IS_nx), end_nx - start_nx, length, end-start])

                print("MAXIMAL-IS IS FOUND AT training step = ", i, "; Cardinality Ours vs. NWx Alg.  = ", [length,len(Maximum_IS_nx)], "; value = ", v_theta.numpy()[0], "; threshold", winning_threshold, "; execution time (sec) = ", end-start)
                #print("MINIMAL VALUE WITH ONES AT THE FOUND MIS IS = ", combined_NN_p(X_MIN_VALUE_TEST), "; (Cardinality Ours(without reductions))^2 = ", [-0.5*(length**2)])
                break

            # logger:
            print('training_step = ', i,  "value = ",
                  v_theta.numpy()[0], ' ; desired value set for optimiztion = ', [P_desired])




############### save results:
print("THIS IS TO CONFIRM THAT THE PROCESS IS FINISHED AND WE ARE SAVING SHIT CORRECTLY")
np.save("/home/user/Desktop/ISMAIL/SSLTL/SSLTL_project/RL_adv_attacks_LP/BOSScombiMIS_2021/Avg_results_OSS_vs_NWx_3.npy",Avg_results_OSS_vs_NWx)

print('break')


# ######################################################################################
# ##################### this the threshold we apply on X_star to get the MIS ###########
# ######################################################################################
# X_star_thresholded = np.zeros(shape=(n_inputss))
# for i in range(n_inputss):
#     if X_star[i] > winning_threshold:
#         X_star_thresholded[i] = 1
#
#
# ##################### IS and Maximal-IS checker:
# Max_IS = MAXIMAL_IS_checker(X_star_thresholded, G, winning_threshold)
# #IS____ =
#
#
# ##################### plot
#
# list_of_nonZeros_nodes = list(np.nonzero(X_star_thresholded)[0])
# # generate node positions:
# pos = nx.spring_layout(G)
# # draw graph
# nx.draw_networkx(G, pos=pos, font_size=16, node_color='blue', font_color='white')
# # draw subgraph for highlights
# nx.draw_networkx(G.subgraph(list_of_nonZeros_nodes), pos=pos, font_size=16, node_color='red', font_color='white')
# plt.draw()
# plt.show()




