import networkx as nx
import numpy as np
import cplex

################################### Load graph to be reduced

G = nx.read_gpickle("graph_to_be_reduced.gpickle")


N_number_of_nodes_org = len(G.nodes)
M_number_of_edges_org = len(G.edges)



# # ################################## plotting the graph with
# nx.draw(G,with_labels=True)
# plt.draw()
# plt.show()


###### Question: When we use LP reduction, do we always end up with a reduced graph thta consists of disjoint subgraphs?
## Answer: not always


components_before = [G.subgraph(c).copy() for c in nx.connected_components(G)]

################################# remove self loops
self_loop_removal_cntr = 0
for pair in list(G.edges):
    if pair[0] == pair[1]:
        G.remove_edge(pair[0],pair[1])
        self_loop_removal_cntr = self_loop_removal_cntr+1
        print("self loop in node {} is removed".format([pair[0]]))
print("ALREADY REMOVED {} EDGES BECAUSE OF SELF LOOP".format(self_loop_removal_cntr))



#############################################################
################### LP ###################
#############################################################
########## INPUT: G
########## set of x_n = 1

problem = cplex.Cplex()

list_of_nodes = list(G.nodes)
list_of_pairs_of_edges = list(G.edges)

### dictionary of id's
node_id = {(n): 'node_id({0})'.format(n) for (n) in list_of_nodes}
problem.objective.set_sense(problem.objective.sense.maximize)
problem.variables.add(names=list(node_id.values()),lb=[0.0]*len(node_id))

## objective:
problem.objective.set_linear(list(zip(list(node_id.values()), [1.0] * len(node_id))))

## constraint: for all (u,v)\in E, node_id(u) + node_id(v) <= 1
""" Constraint (1) """
for (u,v) in list_of_pairs_of_edges:
    #if u != v:
    lin_expr_vars_1 = []
    lin_expr_vals_1 = []
    lin_expr_vars_2 = []
    lin_expr_vals_2 = []
    lin_expr_vars_1.append(node_id[(u)])
    lin_expr_vals_1.append(1.0)
    lin_expr_vars_2.append(node_id[(v)])
    lin_expr_vals_2.append(1.0)
    problem.linear_constraints.add(lin_expr=[
        cplex.SparsePair(lin_expr_vars_1 + lin_expr_vars_2, val=lin_expr_vals_2 + lin_expr_vals_2)],
        rhs=[1.0], senses=["L"],
        names=['(1)_'])


## write program for testing
#problem.write("LP_test.lp")
problem.solve()

if problem.solution.get_solution_type() == 0:
    print("CPLEX is outputting no solution exists")
if problem.solution.get_solution_type() != 0:

    node_id_star = problem.solution.get_values()
    ### removing nodes in (node_id_star == 1) along with their nieghbors
    nodes_tobe_removed = np.where(np.array(node_id_star)==1)

    if len(nodes_tobe_removed[0])==0:
        print("############# LP is solved , BUT without any nodes to be removed ##############")

    nodes_removed_from_LP_reduction=[]
    for node in nodes_tobe_removed[0]:
        # if node is still in graph
        if node in G.nodes:
            # find nieghbors
            nieghbor_list = list(G[int(node)])
            # remove node
            G.remove_node(node)
            nodes_removed_from_LP_reduction.append(node)
            print("LP removing ", node)
            for nie_node in nieghbor_list:
                G.remove_node(nie_node)
                nodes_removed_from_LP_reduction.append(nie_node)
                print("LP removing ", nie_node, "since its a nieghbor of", node)

    if len(G.nodes) == 0:
        print("THIS IS a CASE where a MIS is found with the LP with cardinality = ", np.count_nonzero(node_id_star))


    # re-label reduced graph from 0 to N
    G = nx.relabel.convert_node_labels_to_integers(G)

    ### number to be added to the final MIS of the reduced graph is:


print("NUMBER of nodes in the MIS that we got from the LP", len(nodes_tobe_removed[0]))
nx.write_gpickle(G,"/directory/G_LP_reduced.gpickle")


print("break")




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


#################### get the complimiant of G for the third connections:
#G_hat = nx.complement(G)
#M_number_of_edges_comp = len((G_hat.edges))

N_number_of_nodes = len((G.nodes))
M_number_of_edges = len((G.edges))

M_number_of_edges_comp = (N_number_of_nodes*(N_number_of_nodes-1)/2) - M_number_of_edges

print("Before LP {} - After LP  {}".format([N_number_of_nodes_org, M_number_of_edges_org],[N_number_of_nodes, M_number_of_edges]))


print('break')




components_after = [G.subgraph(c).copy() for c in nx.connected_components(G)]

print("number of dis-joint subgraphs of G =  [Before LP,After LP] ", [len(components_before),len(components_after)])