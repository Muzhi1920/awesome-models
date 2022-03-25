from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math
import re

import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow import feature_column as fc
from tensorflow.python.feature_column import feature_column_v2 as inner_fc


def shared_embedding_columns_v3(categorical_columns,
                                dimension,
                                combiner='mean',
                                initializer=None,
                                shared_embedding_collection_name=None,
                                ckpt_to_load_from=None,
                                tensor_name_in_ckpt=None,
                                max_norm=None,
                                trainable=True,
                                use_safe_embedding_lookup=True):
  """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `shared_embedding_columns` with model_fn:

  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard
      deviation `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional collective name of these columns.
      If not given, a reasonable name will be chosen based on the names of
      `categorical_columns`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies toshaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0.num_buckets
  if not isinstance(c0, inner_fc.CategoricalColumn):
    raise ValueError(
        'All categorical_columns must be subclasses of CategoricalColumn. '
        'Given: {}, of type: {}'.format(c0, type(c0)))
  while isinstance(c0, (inner_fc.WeightedCategoricalColumn, inner_fc.SequenceCategoricalColumn)):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    while isinstance(c, (inner_fc.WeightedCategoricalColumn, inner_fc.SequenceCategoricalColumn)):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same type, or be weighted_categorical_column or sequence column '
          'of the same type. Given column: {} of type: {} does not match given '
          'column: {} of type: {}'.format(c0, type(c0), c, type(c)))
    if num_buckets != c.num_buckets:
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              c0, num_buckets, c, c.num_buckets))

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  column_creator = SharedEmbeddingColumnCreator(
      dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
      num_buckets, trainable, categorical_columns[0], shared_embedding_collection_name,
      use_safe_embedding_lookup)

  result = []
  for column in categorical_columns:
    result.append(
        column_creator(
            categorical_column=column, combiner=combiner, max_norm=max_norm))

  return result

class SharedEmbeddingColumnCreator(tracking.AutoTrackable):

  def __init__(self,
               dimension,
               initializer,
               ckpt_to_load_from,
               tensor_name_in_ckpt,
               num_buckets,
               trainable,
               base_column,
               name='shared_embedding_column_creator',
               use_safe_embedding_lookup=True):
    self._dimension = dimension
    self._initializer = initializer
    self._ckpt_to_load_from = ckpt_to_load_from
    self._tensor_name_in_ckpt = tensor_name_in_ckpt
    self._num_buckets = num_buckets
    self._trainable = trainable
    self._name = name
    self._use_safe_embedding_lookup = use_safe_embedding_lookup
    # Map from graph keys to embedding_weight variables.
    self._embedding_weights = {}
    self._has_built = False
    self._base_column = base_column

  def __call__(self, categorical_column, combiner, max_norm):
    return SharedEmbeddingColumn(self._base_column, categorical_column, self, combiner, max_norm,
                                 self._use_safe_embedding_lookup)

  def get_or_create_embedding_weights(self, base_column, state_manager):
    key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    if key not in self._embedding_weights:
      # print('build embedding weights')
      embedding_shape = (self._num_buckets, self._dimension)
      var = state_manager.create_variable(
        base_column,
        name=self._name,
        shape=embedding_shape,
        dtype=dtypes.float32,
        trainable=self._trainable,
        use_resource=True,
        initializer=self._initializer)
      # print('create share emb name: ', self._name)
      if self._ckpt_to_load_from is not None:
        to_restore = var
        if isinstance(to_restore, variables.PartitionedVariable):
          to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
        checkpoint_utils.init_from_checkpoint(
            self._ckpt_to_load_from, {self._tensor_name_in_ckpt: to_restore})
      self._embedding_weights[key] = var
    else:
      # print('ignore build embedding')
      pass
    return self._embedding_weights[key]

  @property
  def name(self):
    return self._name
  # @property
  # def embedding_weights(self):
  #   key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
  #   if key not in self._embedding_weights:
  #     embedding_shape = (self._num_buckets, self._dimension)
  #     var = variable_scope.get_variable(
  #         name=self._name,
  #         shape=embedding_shape,
  #         dtype=dtypes.float32,
  #         initializer=self._initializer,
  #         trainable=self._trainable)
  #
  #     if self._ckpt_to_load_from is not None:
  #       to_restore = var
  #       if isinstance(to_restore, variables.PartitionedVariable):
  #         to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
  #       checkpoint_utils.init_from_checkpoint(
  #           self._ckpt_to_load_from, {self._tensor_name_in_ckpt: to_restore})
  #     self._embedding_weights[key] = var
  #   return self._embedding_weights[key]

  @property
  def dimension(self):
    return self._dimension


class SharedEmbeddingColumn(
    inner_fc.DenseColumn,
    inner_fc.SequenceDenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    fc_old._SequenceDenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SharedEmbeddingColumn',
        ('base_column', 'categorical_column', 'shared_embedding_column_creator', 'combiner',
         'max_norm', 'use_safe_embedding_lookup'))):
  """See `embedding_column`."""

  def __new__(cls,
              base_column,
              categorical_column,
              shared_embedding_column_creator,
              combiner,
              max_norm,
              use_safe_embedding_lookup=True):
    return super(SharedEmbeddingColumn, cls).__new__(
        cls,
        base_column=base_column,
        categorical_column=categorical_column,
        shared_embedding_column_creator=shared_embedding_column_creator,
        combiner=combiner,
        max_norm=max_norm,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  def create_state(self, state_manager):
    self.shared_embedding_column_creator.get_or_create_embedding_weights(self.base_column, state_manager)

  @property
  def _is_v2_column(self):
    return True

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_shared_embedding'.format(self.categorical_column.name)

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  @property
  def _parse_example_spec(self):
    return inner_fc._raise_shared_embedding_column_error()

  def transform_feature(self, transformation_cache, state_manager):
    """See `FeatureColumn` base class."""
    return transformation_cache.get(self.categorical_column, state_manager)

  def _transform_feature(self, inputs):
    return inner_fc._raise_shared_embedding_column_error()

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape(
        [self.shared_embedding_column_creator.dimension])

  @property
  def _variable_shape(self):
    return inner_fc._raise_shared_embedding_column_error()

  def _get_dense_tensor_internal(self, transformation_cache, state_manager):
    """Private method that follows the signature of _get_dense_tensor."""
    # This method is called from a variable_scope with name _var_scope_name,
    # which is shared among all shared embeddings. Open a name_scope here, so
    # that the ops for different columns have distinct names.
    with ops.name_scope(None, default_name=self.name):
      # Get sparse IDs and weights.
      sparse_tensors = self.categorical_column.get_sparse_tensors(
          transformation_cache, state_manager)
      sparse_ids = sparse_tensors.id_tensor
      sparse_weights = sparse_tensors.weight_tensor
      # print('self info: ', self)
      # print('get share emb name: ', self.shared_embedding_column_creator.name)
      # print('_cols_to_vars_map: ', state_manager._cols_to_vars_map)
      embedding_weights = state_manager.get_variable(self.base_column, name=self.shared_embedding_column_creator.name)
      # print('get finished!')
      sparse_id_rank = tensor_shape.dimension_value(
          sparse_ids.dense_shape.get_shape()[0])
      embedding_lookup_sparse = embedding_ops.safe_embedding_lookup_sparse
      if (not self.use_safe_embedding_lookup and sparse_id_rank is not None and
          sparse_id_rank <= 2):
        embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse_v2
      # Return embedding lookup result.
      return embedding_lookup_sparse(
          embedding_weights,
          sparse_ids,
          sparse_weights,
          combiner=self.combiner,
          name='%s_weights' % self.name,
          max_norm=self.max_norm)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns the embedding lookup result."""
    if isinstance(self.categorical_column, inner_fc.SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must not be of type SequenceCategoricalColumn. '
          'Suggested fix A: If you wish to use DenseFeatures, use a '
          'non-sequence categorical_column_with_*. '
          'Suggested fix B: If you wish to create sequence input, use '
          'SequenceFeatures instead of DenseFeatures. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    return self._get_dense_tensor_internal(transformation_cache, state_manager)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return inner_fc._raise_shared_embedding_column_error()

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """See `SequenceDenseColumn` base class."""
    if not isinstance(self.categorical_column, inner_fc.SequenceCategoricalColumn):
      raise ValueError(
          'In embedding_column: {}. '
          'categorical_column must be of type SequenceCategoricalColumn '
          'to use SequenceFeatures. '
          'Suggested fix: Use one of sequence_categorical_column_with_*. '
          'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                       self.categorical_column))
    dense_tensor = self._get_dense_tensor_internal(transformation_cache,
                                                   state_manager)
    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    sequence_length = fc_utils.sequence_length_from_sparse_tensor(
        sparse_tensors.id_tensor)
    return inner_fc.SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

  def _get_sequence_dense_tensor(self,
                                 inputs,
                                 weight_collections=None,
                                 trainable=None):
    return inner_fc._raise_shared_embedding_column_error()

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]