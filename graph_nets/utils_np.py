# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Auxiliary methods that operate on graph structured data.

This modules contains functions to convert between python data structures
representing graphs and `graphs.GraphsTuple` containing numpy arrays.
In particular:

  - `networkx_to_data_dict` and `data_dict_to_networkx` convert from/to an
    instance of `networkx.OrderedMultiDiGraph` from/to a data dictionary;

  - `networkxs_to_graphs_tuple` and `graphs_tuple_to_networkxs` convert
    from instances of `networkx.OrderedMultiDiGraph` to `graphs.GraphsTuple`;

  - `data_dicts_to_graphs_tuple` and `graphs_tuple_to_data_dicts` convert to and
    from lists of data dictionaries and `graphs.GraphsTuple`;

  - `get_graph` allows to index or slice a `graphs.GraphsTuple` to extract a
    subgraph or a subbatch of graphs.

The functions in these modules are able to deal with graphs containing `None`
fields (e.g. featureless nodes, featureless edges, or no edges).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from graph_nets import graphs
# import networkx as nx
import numpy as np
from six.moves import range
from six.moves import zip  # pylint: disable=redefined-builtin

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS

GRAPH_NX_FEATURES_KEY = "features"


def _check_valid_keys(keys):
  if any([x in keys for x in [EDGES, RECEIVERS, SENDERS]]):
    if not (RECEIVERS in keys and SENDERS in keys):
      raise ValueError("If edges are present, senders and receivers should "
                       "both be defined.")


def _defined_keys(dict_):
  return {k for k, v in dict_.items() if v is not None}


def _check_valid_sets_of_keys(dicts):
  """Checks that all dictionaries have exactly the same valid key sets."""
  prev_keys = None
  for dict_ in dicts:
    current_keys = _defined_keys(dict_)
    _check_valid_keys(current_keys)
    if prev_keys and current_keys != prev_keys:
      raise ValueError(
          "Different set of keys found when iterating over data dictionaries "
          "({} vs {})".format(prev_keys, current_keys))
    prev_keys = current_keys


def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked np arrays.

  When a set of np arrays are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked np array. This
  computes those offsets.

  Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.

  Returns:
    The index offset per graph.
  """
  return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def data_dicts_to_graphs_tuple(data_dicts):
  """Constructs a `graphs.GraphsTuple` from an iterable of data dicts.

  The graphs represented by the `data_dicts` argument are batched to form a
  single instance of `graphs.GraphsTuple` containing numpy arrays.

  Args:
    data_dicts: An iterable of dictionaries with keys `GRAPH_DATA_FIELDS`, plus,
      potentially, a subset of `GRAPH_NUMBER_FIELDS`. The NODES and EDGES fields
      should be numpy arrays of rank at least 2, while the RECEIVERS, SENDERS
      are numpy arrays of rank 1 and same dimension as the EDGES field first
      dimension. The GLOBALS field is a numpy array of rank at least 1.

  Returns:
    An instance of `graphs.GraphsTuple` containing numpy arrays. The
    `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to `np.int32`
    type.
  """
  for key in graphs.GRAPH_DATA_FIELDS:
    for data_dict in data_dicts:
      data_dict.setdefault(key, None)
  _check_valid_sets_of_keys(data_dicts)
  data_dicts = _to_compatible_data_dicts(data_dicts)
  return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))


def graphs_tuple_to_data_dicts(graph):
  """Splits the stored data into a list of individual data dicts.

  Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
  SENDERS.

  Args:
    graph: A `graphs.GraphsTuple` instance containing numpy arrays.

  Returns:
    A list of the graph data dictionaries. The GLOBALS field is a tensor of
      rank at least 1, as the RECEIVERS and SENDERS field (which have integer
      values). The NODES and EDGES fields have rank at least 2.
  """
  offset = _compute_stacked_offsets(graph.n_node, graph.n_edge)

  nodes_splits = np.cumsum(graph.n_node[:-1])
  edges_splits = np.cumsum(graph.n_edge[:-1])
  graph_of_lists = collections.defaultdict(lambda: [])
  if graph.nodes is not None:
    graph_of_lists[NODES] = np.split(graph.nodes, nodes_splits)
  if graph.edges is not None:
    graph_of_lists[EDGES] = np.split(graph.edges, edges_splits)
  if graph.receivers is not None:
    graph_of_lists[RECEIVERS] = np.split(graph.receivers - offset, edges_splits)
    graph_of_lists[SENDERS] = np.split(graph.senders - offset, edges_splits)
  if graph.globals is not None:
    graph_of_lists[GLOBALS] = _unstack(graph.globals)

  n_graphs = graph.n_node.shape[0]
  # Make all fields the same length.
  for k in GRAPH_DATA_FIELDS:
    graph_of_lists[k] += [None] * (n_graphs - len(graph_of_lists[k]))
  graph_of_lists[N_NODE] = graph.n_node
  graph_of_lists[N_EDGE] = graph.n_edge

  result = []
  for index in range(n_graphs):
    result.append({field: graph_of_lists[field][index] for field in ALL_FIELDS})
  return result


def _to_compatible_data_dicts(data_dicts):
  """Converts the content of `data_dicts` to arrays of the right type.

  All fields are converted to numpy arrays. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `np.int32`.

  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and values
      either `None`s, or quantities that can be converted to numpy arrays.

  Returns:
    A list of dictionaries containing numpy arrays or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:
        dtype = np.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        result[k] = np.asarray(v, dtype)
    results.append(result)
  return results


def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-None NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-None RECEIVERS field;
  otherwise, it is set to 0.

  Args:
    data_dict: An input `dict`.

  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = np.array(
            np.shape(dct[data_field])[0], dtype=np.int32)
      else:
        dct[number_field] = np.array(0, dtype=np.int32)
  return dct


def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.

  Args:
    data_dicts: An iterable of data dictionaries with keys `GRAPH_DATA_FIELDS`,
      plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`. Each dictionary is
      representing a single graph.

  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.
  """
  # Create a single dict with fields that contain sequences of graph tensors.
  concatenated_dicts = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        concatenated_dicts[k].append(v)
      else:
        concatenated_dicts[k] = None

  concatenated_dicts = dict(concatenated_dicts)

  for field, arrays in concatenated_dicts.items():
    if arrays is None:
      concatenated_dicts[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      concatenated_dicts[field] = np.stack(arrays)
    else:
      concatenated_dicts[field] = np.concatenate(arrays, axis=0)

  if concatenated_dicts[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(concatenated_dicts[N_NODE],
                                      concatenated_dicts[N_EDGE])
    for field in (RECEIVERS, SENDERS):
      concatenated_dicts[field] += offset

  return concatenated_dicts


def get_graph(input_graphs, index):
  """Indexes into a graph.

  Given a `graphs.GraphsTuple` containing arrays and an index (either
  an `int` or a `slice`), index into the nodes, edges and globals to extract the
  graphs specified by the slice, and returns them into an another instance of a
  `graphs.GraphsTuple` containing `Tensor`s.

  Args:
    input_graphs: A `graphs.GraphsTuple` containing numpy arrays.
    index: An `int` or a `slice`, to index into `graph`. `index` should be
      compatible with the number of graphs in `graphs`.

  Returns:
    A `graphs.GraphsTuple` containing numpy arrays, made of the extracted
      graph(s).

  Raises:
    TypeError: if `index` is not an `int` or a `slice`.
  """
  if isinstance(index, int):
    graph_slice = slice(index, index + 1)
  elif isinstance(index, slice):
    graph_slice = index
  else:
    raise TypeError("unsupported type: %s" % type(index))
  data_dicts = graphs_tuple_to_data_dicts(input_graphs)[graph_slice]
  return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))
