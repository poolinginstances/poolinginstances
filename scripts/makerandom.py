#! /usr/bin/env python2

# Script to randomly generate Pooling instances based on copies of the
# Haverly networks. Run "makerandom.py -h" to get documentation.
#
# Requires the networkx library to be installed. Tested only with python 2.7.

import json
import itertools

import networkx as nx
from networkx.readwrite import json_graph


def get_haverly(haverly_idx, node_prefix=None):
    """Returns Haverly graph as nx.DiGraph

    Parameters:
    - haverly_idx: There are 3 graphs in Haverlys paper. This
    argument gives the index of the graph that is desired
    - node_prefix: A prefix for all nodes
    """

    G = nx.DiGraph(attributes=["k1"])

    # add the nodes and store its type in the "type" attribute
    G.add_nodes_from(["i1", "i2", "i3"], type="input")
    G.add_nodes_from(["l1"], type="pool")
    G.add_nodes_from(["j1", "j2"], type="output")

    # Store the capacities in the "C" attribute
    if haverly_idx in [1, 3]:
        G.node["l1"]["C"] = 300
        G.node["j1"]["C"] = 100
        G.node["j2"]["C"] = 200
        input_capacity = 300
    else:
        G.node["l1"]["C"] = 800
        G.node["j1"]["C"] = 600
        G.node["j2"]["C"] = 200
        input_capacity = 800
    for n in ["i1", "i2", "i3"]:
        G.node[n]["C"] = input_capacity

    # Add input quality in the "lambda" attibute. The "lambda"
    # attribute is a dictionary as there might be several qualities to
    # be tracked.
    G.node["i1"]["lambda"] = {"k1": 3.0}
    G.node["i2"]["lambda"] = {"k1": 1.0}
    G.node["i3"]["lambda"] = {"k1": 2.0}

    # Add upper quality bounds at outputs
    G.node["j1"]["overbeta"] = {"k1": 2.5}
    G.node["j2"]["overbeta"] = {"k1": 1.5}

    # finally add the edges with its cost
    G.add_edges_from([("i1", "l1", {"cost": 6}),
                      ("i2", "l1", {"cost": 13 if haverly_idx == 3 else 16}),
                      ("l1", "j1", {"cost": -9}),
                      ("l1", "j2", {"cost": -15}),
                      ("i3", "j1", {"cost": 1}),
                      ("i3", "j2", {"cost": -5})])

    # rename the nodes to include the prefix
    if node_prefix:
        nx.relabel_nodes(G, lambda v: "%s_%s" % (node_prefix, v), copy=False)

    return G

def add_attribute(G, num):
    """ Add randomly sampled attributes

    Parameters:
    - G: The graph to add the attributes to
    - num: The number of attribues to be added
    """
    if not num:
        return

    for i in range(num):
        name = "ka_%d" % i
        G.graph["attributes"].append(name)

        # Sample input qualities uniformly between 0 and 10
        for v in get_inputs(G):
            G.node[v]["lambda"][name] = random.uniform(0.0, 10.0)

        # Sample output upper quality bound uniformly between the min
        # and max input quality of reachable inputs. Values out of
        # this range would be either not attainable or not a
        # restriction
        for v in get_outputs(G):
            # sample only from the range of the reachable inputs
            inp = [i for (i, l) in itertools.product(get_inputs(G), get_pools(G)) if G.has_edge(i, l) and G.has_edge(l, v)]
            inp += [i for i in get_inputs(G) if G.has_edge(i, v)]
            minlambda = min(G.node[i]["lambda"][name] for i in inp)
            maxlambda = max(G.node[i]["lambda"][name] for i in inp)

            G.node[v]["overbeta"][name] = random.uniform(minlambda, maxlambda)

def add_edge(G, candidates, mincost, maxcost):
    """Add one edge from a candidate set

    If the graph is not connected, we select an edge that connects to
    components.

    Parameters:
    - G: The graph to add the edges to

    - candidates: The "set" of candidate edges (the type needs to be
      compatible with random.choice and has to have a remove method)

    - mincost: Minimum cost for the edge

    - maxcost: Maximum cost for the edge
    """

    cost = random.uniform(mincost, maxcost)
    components = nx.number_connected_components(G.to_undirected())
    u, v = random.choice(candidates)

    while components > 1:
        # if we have more than one components, we want an edge that
        # decreases the number of connected components. Therefore we
        # copy the graph and add the edge to the copy.
        G_copy = G.copy()
        G_copy.add_edge(u, v)

        #if the number of if components decreased, go out of the loop.
        #Otherwise sample new u, v
        if nx.number_connected_components(G_copy.to_undirected()) < components:
            break
        else:
            u, v = random.choice(candidates)

    # finally add the edge and remove it from the candidates
    G.add_edge(u, v, cost=cost)
    candidates.remove((u, v))


def add_edges(G, density=None, num=None):
    """Add edges to graph

    Either a target density or a number of edges must be given, but
    not both. If the graph is not connected, an edge that connects two
    components is selected.

    Parameters:
    - G: The graph to add the edges to

    - density: The target density. Edges are added until the density
      is >= this value. The density is computed wrt. the admissible
      edges in a pooling instances (edges are only allowed between
      inputs-pools, pool-outputs and inputs-outputs

    - num: Fixed number of nodes to be added
    """
    assert(density is None or num is None)

    candidates = nx.complement(G).edges()
    candidates = [(u, v) for u, v in candidates
                  if ((G.node[u]["type"] == "input" and
                       G.node[v]["type"] == "pool")
                      or
                      (G.node[u]["type"] == "pool" and
                       G.node[v]["type"] == "output")
                      or
                      (G.node[u]["type"] == "input" and
                       G.node[v]["type"] == "output"))]

    mincost = get_min_costs(G)
    maxcost = get_max_costs(G)

    if density is not None:
        while get_density(G) < density:
            add_edge(G, candidates, mincost, maxcost)

    if num is not None:
        for _ in range(num):
            add_edge(G, candidates, mincost, maxcost)

def get_inputs(G):
    """ Return list of inputs """
    return [n for n in G.nodes() if G.node[n]["type"] == "input"]

def get_pools(G):
    """ Return list of pools """
    return [n for n in G.nodes() if G.node[n]["type"] == "pool"]

def get_outputs(G):
    """ Return list of output """
    return [n for n in G.nodes() if G.node[n]["type"] == "output"]

def get_edges_input_to_pool(G):
    """ Return edges from inputs to pools"""
    return [(u, v) for u, v in G.edges() if G.node[u]["type"] == "input" and G.node[v]["type"] == "pool"]

def get_edges_pool_to_output(G):
    """ Return edges from pools to outputs"""
    return [(u, v) for u, v in G.edges() if G.node[u]["type"] == "pool" and G.node[v]["type"] == "output"]

def get_edges_input_to_output(G):
    """ Return edges from inputs to outputs"""
    return [(u, v) for u, v in G.edges() if G.node[u]["type"] == "input" and G.node[v]["type"] == "output"]

def get_min_costs(G):
    """ Return minimum edges cost in the graph"""

    return min(G.edge[i][j]["cost"] for i, j in G.edges())

def get_max_costs(G):
    """ Return maximum edges cost in the graph"""
    return max(G.edge[i][j]["cost"] for i, j in G.edges())

def get_density(G):
    """ compute the density as the number of edges divided by the number of possible edges."""
    ni = len(get_inputs(G))
    no = len(get_outputs(G))
    np = len(get_pools(G))
    return float(nx.number_of_edges(G)) / float(ni*np + np*no + ni*no)

def get_attributes(G):
    """ Return the list of quality attributes"""
    return G.graph["attributes"]

def check_graph(G):
    """ Check consistency of the pooling graph """
    assert "attributes" in G.graph
    assert(len(get_inputs(G)) > 0)
    assert(len(get_pools(G)) > 0)
    assert(len(get_outputs(G)) > 0)
    assert(all("type" in G.node[n] for n in G.nodes()))
    assert(all((G.node[n]["type"] in ["input", "output", "pool"]) for n in G.nodes()))
    assert(all("C" in G.node[n] for n in G.nodes()))
    assert(all("lambda" in G.node[n] for n in get_inputs(G)))
    assert(all(all((k in G.node[n]["lambda"]) for k in G.graph["attributes"]) for n in get_inputs(G)))

    assert(all(all((k in G.node[n].get("overbeta", [])) or (k in G.node[n].get("underbeta", [])) for k in G.graph["attributes"]) for n in get_outputs(G)))

    assert(nx.number_of_edges(G) == len(get_edges_input_to_pool(G)) + len(get_edges_pool_to_output(G)) + len(get_edges_input_to_output(G)))

def get_stats(G):
    """ Return a dict with statistics about the graph"""
    from collections import OrderedDict

    stats = OrderedDict()

    stats["nodes"] = nx.number_of_nodes(G)
    stats["edges"]  = nx.number_of_edges(G)
    stats["inputs"] = len(get_inputs(G))
    stats["pools"] = len(get_pools(G))
    stats["outputs"] = len(get_outputs(G))
    stats["components"] = nx.number_connected_components(G.to_undirected())
    stats["density_real"] = get_density(G)
    stats["input->pool"] = len(get_edges_input_to_pool(G))
    stats["pool->output"] = len(get_edges_pool_to_output(G))
    stats["input->output"] = len(get_edges_input_to_output(G))
    stats["min_cost"] = get_min_costs(G)
    stats["max_cost"] = get_max_costs(G)



    return stats

def export_json(G, args, filename):
    """Export the graph with the scripts args to json file

    Parameters:
    - G: The graph to be exported

    - args: The args of the call to the script. Usually the output of
            parser.parse_args() where parser is an ArgumentParser

    - filename: The filename of the json file
    """

    data = vars(args)
    data.update(get_stats(G))

    data["graph"] = json_graph.node_link_data(G)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def export_to_GAMS(G, args, filename):
    """Export the graph to GAMS file

    The arguments of the script and some statistics about the graph
    will be included as comment for documentation.

    Parameters:
    - G: The graph to be exported

    - args: The args of the call to the script. Usually the output of
            parser.parse_args() where parser is an ArgumentParser

    - filename: The filename of the GAMS file
    """


    check_graph(G)
    stats = get_stats(G)

    with open(filename, "w") as f:

        f.write("* Generated with the following parameters:\n")
        maxkeylen = max(map(len,  vars(args).keys()))
        fmt = "* \t%%-%ds : %%s\n" % maxkeylen
        for key, val in vars(args).iteritems():
            f.write(fmt % (key, val))
        f.write("\n")


        f.write("* Statistics about the input graph:\n")
        maxkeylen = max(map(len, stats.keys()))
        fmt = "* \t%%-%ds : %%s\n" % maxkeylen
        for key, val in stats.iteritems():
            f.write(fmt % (key, val))
        f.write("\n")

        f.write("set V /%s/;\n" % ", ".join(G.nodes()))
        f.write("set K /%s/;\n" % ", ".join(get_attributes(G)))

        f.write("set I /%s/;\n" % ", ".join(get_inputs(G)))
        f.write("set L /%s/;\n" % ", ".join(get_pools(G)))
        f.write("set J /%s/;\n" % ", ".join(get_outputs(G)))

        f.write("set A /\n%s\n/;\n" % ", \n".join("    %s.%s" % (i, j) for (i, j) in G.edges()))
        f.write("parameter cost(V,V) /\n%s\n/;\n" % "\n".join("    %s.%s %f" % (i, j, G.edge[i][j].get("cost", 0.0)) for (i, j) in G.edges()))
        f.write("parameter C(V) /\n%s\n/;\n" % "\n".join("    %s %f" % (i, G.node[i]["C"]) for i in G.nodes()))
        f.write("parameter lambda(K,V) /\n%s\n/;\n" % "\n".join(
                    "    %s.%s %f" % (k, i, G.node[i]["lambda"][k]) for k in get_attributes(G) for i in get_inputs(G)))
        f.write("parameter overbeta(K,V) /\n%s\n/;\n" % "\n".join(
                    "    %s.%s %s" % (k, i, G.node[i].get("overbeta", {}).get(k, "INF")) for k in get_attributes(G) for i in get_outputs(G)))
        f.write("parameter underbeta(K,V) /\n%s\n/;\n" % "\n".join(
                    "    %s.%s %s" % (k, i, G.node[i].get("underbeta", {}).get(k, "-INF")) for k in get_attributes(G) for i in get_outputs(G)))
        f.write("parameter ub(V,V) /\n%s\n/;\n" % "\n".join(
                    "    %s.%s %s" % (i, j, G.edge[i][j].get("ub", "INF")) for i, j in G.edges()))
        f.write("A1(I,L) = yes$(A(I,L)) ;\n")
        f.write("A2(L,J) = yes$(A(L,J)) ;\n")
        f.write("A3(I,J) = yes$(A(I,J)) ;\n")
        f.write("T(I,L,J) = yes$(A1(I,L) and A2(L,J)) ;\n")


def random_factor(fac_min=0.25, fac_max=4.0):
    """when we want factors between e.g. 0.25 and 4, we get many more values > 1 than <1.

    Therefore we define a function f which maps the range [-1, 1] onto
    [fac_min, fac_max] with fac_min < 1, fac_max > 1 such that
    f(-1) = fac_min
    f( 1) = fac_max
    f( 0) = 1

    f therefore transforms a uniform distribution into something that is >0 or <0 with equal proability.
    """

    assert fac_min < 1
    assert fac_max > 1

    a = 0.5 * (fac_max + fac_min - 2.0)
    b = 0.5 * (fac_max - fac_min)
    f = lambda z : a*z**2 + b*z + 1

    z = random.uniform(-1.0, 1.0)
    return f(z)

if __name__ == "__main__":

    import argparse
    import random

    parser = argparse.ArgumentParser(description='Generate some random pooling instances')
    parser.add_argument("out", type=str, help="The output file name")
    parser.add_argument("--seed", type=int, help="Random seed to be used")
    parser.add_argument('--haverly', type=int, help='Number of copies of the Haverly network to start with')
    parser.add_argument('--scalehaverlys', help='Specify how the quality is scaled of the different Haverly graphs.')
    parser.add_argument('--scalequalities', help='Specify the fraction of inputs that get their lambda changed.')
    parser.add_argument('--density', type=float, help='The targeted density.')
    parser.add_argument('--addedges', type=int, help='Add specific number of edges.')
    parser.add_argument('--attributes', type=int, help='Number of additional attributes.')


    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 99999999)
    random.seed(args.seed)

    # setup empty graph. Additional components can be added or composed into it
    G = nx.DiGraph()

    # add the desired number of copies of the Haverly graph
    if args.haverly >= 1:
        prefix_fmt = "h%%0%dd" % len(str(args.haverly-1))


        for n in xrange(args.haverly):
            # Scale all qualities with a random factor to not have all
            # copies of the Haverly network identical
            factor = 1.0
            if args.scalehaverlys == "random":
                factor = random_factor(0.5, 2.0)
            K = get_haverly(random.randint(1, 3), node_prefix=prefix_fmt % n)
            for i in get_inputs(K):
                for k in get_attributes(K):
                    K.node[i]["lambda"][k] *= factor
            for j in get_outputs(K):
                for k in K.node[j].get("overbeta", {}):
                    K.node[j]["overbeta"][k] *= factor
                for k in K.node[j].get("underbeta", {}):
                    K.node[j]["underbeta"][k] *= factor

            G = nx.compose(G, K)

    if args.density:
        add_edges(G, density=args.density)
    if args.addedges:
        add_edges(G, num=args.addedges)

    if args.scalequalities:
        inputs = get_inputs(G)
        nchanges = int(float(args.scalequalities)*len(inputs))
        args.scalaquality_entries = nchanges
        toscale = random.sample(inputs, nchanges)
        for i in toscale:
            for k in get_attributes(G):
                G.node[i]["lambda"][k] *= random_factor(0.25, 4.0)

    add_attribute(G, args.attributes)

    # export the constructed graph
    export_to_GAMS(G, args, "%s.dat" % args.out)
    export_json(G, args, "%s.json" % args.out)
