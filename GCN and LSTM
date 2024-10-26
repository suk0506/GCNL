

"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras import ops
import time
# import keras
print(tf.__version__)
print(keras.__version__)

"""
## Data preparation
"""

"""
### Data description

We use a real-world traffic speed dataset named `PeMSD7`. We use the version
collected and prepared by [Yu et al., 2018](https://arxiv.org/abs/1709.04875)
and available
[here](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/dataset).

The data consists of two files:

- `PeMSD7_W_228.csv` contains the distances between 228
stations across the District 7 of California.
- `PeMSD7_V_228.csv` contains traffic
speed collected for those stations in the weekdays of May and June of 2012.

The full description of the dataset can be found in
[Yu et al., 2018](https://arxiv.org/abs/1709.04875).
"""

"""
### Loading data
"""

# url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip"
# data_dir = keras.utils.get_file(origin=url, extract=True, archive_format="zip")
# data_dir = data_dir.rstrip("PeMSD7_Full.zip")




def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    # test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, mean, std



def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):

    inputs = keras.utils.timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = keras.utils.timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()



# test_dataset = create_tf_dataset(
#     test_array,
#     input_sequence_length,
#     forecast_horizon,
#     batch_size=test_array.shape[0],
#     shuffle=False,
#     multi_horizon=multi_horizon,
# )



def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


"""
The function `compute_adjacency_matrix()` returns a boolean adjacency matrix
where 1 means there is an edge between two nodes. We use the following class
to store the information about the graph.
"""


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes



class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = self.add_weight(
            initializer=keras.initializers.GlorotUniform(),
            shape=(in_feat, out_feat),
            dtype="float32",
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return ops.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return ops.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = ops.concatenate([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")
        return self.activation(h)

    def call(self, features):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)



class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(self, in_feat, out_feat, model_names, lstm_units: int, input_seq_len: int, output_seq_len: int,
        graph_info: GraphInfo, graph_conv_params: typing.Optional[dict] = None, **kwargs,):

        super().__init__(**kwargs)
            # 类型检查
        if graph_conv_params is not None and not isinstance(graph_conv_params, dict):
            raise TypeError("graph_conv_params must be a dictionary")


        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "add",
                "activation": None,
            }

        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)
        if model_names == 1:

            self.lstm = layers.LSTM(lstm_units, activation="sigmoid")

        elif model_names == 2:

            self.lstm = layers.GRU(lstm_units, activation="sigmoid")


        # self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)
        # # self.lstm1 = layers.GRU(lstm_units, activation="relu", return_sequences=True)
        # # self.lstm = layers.Bidirectional(layers.GRU(lstm_units, activation="relu"))
        # # self.dense1 = layers.Dense(output_seq_len)
        # # self.lstm1 = layers.Bidirectional(layers.LSTM(lstm_units, activation="relu", return_sequences=True))
        # # self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=lstm_units)
        # # self.lstm = layers.Bidirectional(layers.LSTM(lstm_units, activation="relu"))
        # self.lstm = layers.LSTM(lstm_units, activation="sigmoid")
        # self.lstm = layers.GRU(lstm_units, activation="sigmoid")
        # # self.lstm = layers.Bidirectional(layers.LSTM(lstm_units, activation="relu"))
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

        Returns:
            A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
        """

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = ops.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = ops.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = ops.reshape(
            gcn_out, (batch_size * num_nodes, input_seq_len, out_feat)
        )
        # lstm1_out = self.lstm1(
        #     gcn_out
        # )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)
        # dense1_output = self.dense(
        #     lstm1_out
        # )
        # bilstm_out = self.bilstm(lstm_out)
        # Self-Attention Mechanism
        # attention_out = self.attention(lstm1_out, lstm1_out)  # attention_out has shape: (batch_size * num_nodes, input_seq_len, lstm_units)

        # BiLSTM layer
        # bilstm_out = self.bilstm(attention_out)  # bilstm_out has shape: (batch_size * num_nodes, 2 * bilstm_units)

        lstm_out = self.lstm(
            gcn_out
        )
        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = ops.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return ops.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)


for file in ['DS1.xlsx','DS2.xlsx','DS3.xlsx']:
    
    model_names = 2
    speeds_array = pd.read_excel(os.path.join(file), header=None).to_numpy()

    my_list = list(range(80))
    # my_list = my_list[:805]
    print(my_list)
    sample_routes = my_list

    # print(sample_routes)
    # route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
    speeds_array = speeds_array[:, sample_routes]


    # train_size, val_size = 0.5, 0.2
    # train_array, val_array,  mean1, std1 = preprocess(speeds_array, train_size, val_size)

    # print(f"train set size: {train_array.shape}")
    # print(f"validation set size: {val_array.shape}")
    # # print(f"test set size: {test_array.shape}")

    # in_feat = 1
    # batch_size = 32
    # epochs = 200
    # input_sequence_length = 12
    # forecast_horizon = 5
    # # multi_horizon = False
    # multi_horizon = True
    # out_feat = 10
    # lstm_units = 128
    

    # train_dataset, val_dataset = (
    #     create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    #     for data_array in [train_array, val_array]
    # )


    # sigma2 = 0.1
    # epsilon = 0.5
    # route_distances = pd.read_excel(os.path.join("CS_edge2.xlsx"), header=None).to_numpy()
    # adjacency_matrix = route_distances
    # # adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)

    # # print(f"adjacency_matrix shape={adjacency_matrix.shape}")
    # # print(f"adjacency_matrix={adjacency_matrix}")

    # node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    # print(node_indices.shape, neighbor_indices.shape)
    # graph = GraphInfo(
    #     edges=(node_indices.tolist(), neighbor_indices.tolist()),
    #     num_nodes=adjacency_matrix.shape[0],
    # )
    # print(f"number of nodes: {graph.num_nodes}, number of edges: {graph.edges}")

    # graph_conv_params = {
    #     "aggregation_type": "mean",
    #     "combination_type": "add",
    #     "activation": None,
    # }

    # st_gcn = LSTMGC(
    #     in_feat,
    #     out_feat,
    #     model_names,
    #     lstm_units,
    #     input_sequence_length,
#     forecast_horizon,


#     graph,


#     graph_conv_params,


#     # model_names,


    #         )
# if model_names == 1:


#     moded = "LSTM"


# elif model_names == 2:


#     moded = "GRU"



# time0 = time.time()


# for i in range(20):


        
#     inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))


#     # print(inputs)


#     outputs = st_gcn(inputs)



#     model = keras.models.Model(inputs, outputs)


#     model.compile(


#         optimizer=keras.optimizers.RMSprop(learning_rate=0.002),


#         loss=keras.losses.MeanSquaredError(),


    #     )
#     model.fit(


#         train_dataset,


#         validation_data=val_dataset,


#         epochs=epochs,


#         verbose=1,


#         callbacks=[keras.callbacks.EarlyStopping(patience=10)],


    #     )

#     file1 = file.split('.')[0]



#     tf.saved_model.save(model, f"{file1}_saved_model_daily_WL1\\{moded}{i}_sigmod_add_RMSprop_lr0.002_128u_12-5horizon")


    #     """
#     ## Making forecasts on test set



#     Now we can use the trained model to make forecasts for the test set. Below, we


#     compute the MAE of the model and compare it to the MAE of naive forecasts.


#     The naive forecasts are the last value of the speed for each node.


    #     """
# time1 = time.time() - time0


# print(f"{moded}{file1}time: {time1}") 
