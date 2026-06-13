import numpy as np
from scipy import sparse
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, to_dense_batch

import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch.jit as jit
from .util import init, get_clones
import torch.nn.functional as F


class EmbedConv(MessagePassing):
    """
    EmbedConv 类定义与代码1保持一致，无需修改。
    """
    def __init__(self, 
                input_dim:int, 
                num_embeddings:int, 
                embedding_size:int, 
                hidden_size:int,
                layer_N:int,
                use_orthogonal:bool,
                use_ReLU:bool,
                use_layerNorm:bool,
                add_self_loop:bool,
                edge_dim:int=0):
        super(EmbedConv, self).__init__(aggr='add')
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity()
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)

        # 定义第一层线性层
        self.lin1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)
        
        # 初始化隐藏层
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)
        
        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x:Union[Tensor, OptPairTensor], edge_index:Adj, edge_attr:OptTensor=None):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j:Tensor, edge_attr:OptTensor):
        node_feat_j = x_j[:, :-1]
        entity_type_j = x_j[:, -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        x = self.active_func(x)
        x = self.layer_norm(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class SimplifiedAttentionNet(nn.Module):
    """
    实现 Communication Enhanced Network (CEN)
    使用多头图注意力机制（Graph Attention）和软注意力门（Soft Attention Gate）。
    """
    def __init__(self,
                input_dim: int,
                num_embeddings: int,
                embedding_size: int,
                hidden_size: int,
                layer_N: int,
                use_ReLU: bool,
                graph_aggr: str,
                global_aggr_type: str,
                embed_hidden_size: int,
                embed_layer_N: int,
                embed_use_orthogonal: bool,
                embed_use_ReLU: bool,
                embed_use_layerNorm: bool,
                embed_add_self_loop: bool,
                max_edge_dist: float,
                edge_dim: int = 1,
                num_heads: int = 3):  # 新增参数：多头注意力头数
        super(SimplifiedAttentionNet, self).__init__()
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        self.num_heads = num_heads  # 多头数量

        # 嵌入层
        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,  # 减1是因为 node_obs = [node_feat, entity_type]
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )

        # 多头注意力机制
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_hidden_size, hidden_size),
                self.active_func,
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_heads)
        ])

        # 软注意力门的权重计算模块
        self.soft_attention_gate = nn.Sequential(
            nn.Linear(embed_hidden_size, hidden_size),
            self.active_func,
            nn.Linear(hidden_size, num_heads)
        )

    def forward(self, batch: Batch, agent_id: Tensor):
        """
        Args:
            batch (Batch): 包含节点特征、边索引和边属性的批量图数据。
            agent_id (Tensor): 特定节点的索引，用于提取特定节点的特征。

        Returns:
            Tensor: 聚合后的图特征或者节点特定特征。
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.embed_layer(x, edge_index, edge_attr)

        # 多头注意力机制：计算每个头的特征
        head_outputs = [head(x) for head in self.attention_heads]  # List[Tensor]
        head_outputs = torch.stack(head_outputs, dim=-1)  # Shape: [num_nodes, hidden_size, num_heads]

        # 软注意力门：计算每个头的权重
        alpha = self.soft_attention_gate(x)  # Shape: [num_nodes, num_heads]
        alpha = F.softmax(alpha, dim=-1)  # 对每个头的权重进行归一化

        # 聚合多头输出
        x = (head_outputs * alpha.unsqueeze(1)).sum(dim=-1)  # Shape: [num_nodes, hidden_size]

        # 如果需要，转换为稠密批量
        x, mask = to_dense_batch(x, batch.batch)

        if self.graph_aggr == "node":
            return self.gatherNodeFeats(x, agent_id)  # 直接传入 agent_id
        elif self.graph_aggr == "global":
            return self.graphAggr(x)

        raise ValueError(f"Invalid graph_aggr: {self.graph_aggr}")

    @staticmethod
    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        """
        Process adjacency matrix to filter far away nodes
        and then obtain the edge_index and edge_weight
        `adj` is of shape (batch_size, num_nodes, num_nodes)
        OR (num_nodes, num_nodes)
        """
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        # filter far away nodes and connection to itself
        connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
        adj = adj * connect_mask
        if adj.dim() == 3:
            # Case: (batch_size, num_nodes, num_nodes)
            batch_size, num_nodes, _ = adj.shape
            edge_index = adj.nonzero(as_tuple=False)
            edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
            # Adjust indices for batched graph
            batch = edge_index[:, 0] * num_nodes
            edge_index = torch.stack([batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0)
        else:
            # Case: (num_nodes, num_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]]

        # Ensure edge_attr is 2D
        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr

        return edge_index, edge_attr

    def gatherNodeFeats(self, x: Tensor, idx: Tensor):
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)  # (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats).unsqueeze(1)  # (batch_size, 1, num_feats)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)  # (batch_size, num_feats)
            out.append(gathered_node)
        return torch.cat(out, dim=1)  # (batch_size, num_feats*k)

    def graphAggr(self, x: Tensor):
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            return x.max(dim=1)[0]
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"Invalid global_aggr_type: {self.global_aggr_type}")


class GNNBase(nn.Module):
    """
        A Wrapper for constructing the Base graph neural network.
        This uses TransformerConv from Pytorch Geometric
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
        and embedding layers for entity types
        Params:
        args: (argparse.Namespace)
            Should contain the following arguments
            num_embeddings: (int)
                Number of entity types in the env to have different embeddings 
                for each entity type
            embedding_size: (int)
                Embedding layer output size for each entity category
            embed_hidden_size: (int)
                Hidden layer dimension after the embedding layer
            embed_layer_N: (int)
                Number of hidden linear layers after the embedding layer")
            embed_use_ReLU: (bool)
                Whether to use ReLU in the linear layers after the embedding layer
            embed_add_self_loop: (bool)
                Whether to add self loops in adjacency matrix
            gnn_hidden_size: (int)
                Hidden layer dimension in the GNN
            gnn_num_heads: (int)
                Number of heads in the transformer conv layer (GNN)
            gnn_concat_heads: (bool)
                Whether to concatenate the head output or average
            gnn_layer_N: (int)
                Number of GNN conv layers
            gnn_use_ReLU: (bool)
                Whether to use ReLU in GNN conv layers
            max_edge_dist: (float)
                Maximum distance above which edges cannot be connected between 
                the entities
            graph_feat_type: (str)
                Whether to use 'global' node/edge feats or 'relative'
                choices=['global', 'relative']
        node_obs_shape: (Union[Tuple, List])
            The node observation shape. Example: (18,)
        edge_dim: (int)
            Dimensionality of edge attributes 
    """
    def __init__(self, args:argparse.Namespace, 
                node_obs_shape:Union[List, Tuple],
                edge_dim:int, graph_aggr:str):
        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads

        self.gnn = SimplifiedAttentionNet(input_dim=node_obs_shape, edge_dim=edge_dim,
                    num_embeddings=args.num_embeddings,
                    embedding_size=args.embedding_size,
                    hidden_size=args.gnn_hidden_size,
                    layer_N=args.gnn_layer_N,
                    use_ReLU=args.gnn_use_ReLU,
                    graph_aggr=graph_aggr,
                    global_aggr_type=args.global_aggr_type,
                    embed_hidden_size=args.embed_hidden_size,
                    embed_layer_N=args.embed_layer_N,
                    embed_use_orthogonal=args.use_orthogonal,
                    embed_use_ReLU=args.embed_use_ReLU,
                    embed_use_layerNorm=args.use_feature_normalization,
                    embed_add_self_loop=args.embed_add_self_loop,
                    max_edge_dist=args.max_edge_dist,
                    num_heads=args.gnn_num_heads)
        self.out_dim = args.gnn_hidden_size * (args.gnn_num_heads if args.gnn_concat_heads else 1)
        
    def forward(self, node_obs:Tensor, adj:Tensor, agent_id:Tensor):
        batch_size, num_nodes, _ = node_obs.shape
        edge_index, edge_attr = SimplifiedAttentionNet.process_adj(adj, self.gnn.max_edge_dist)
        # print("Outer edge_index", edge_index.shape, "node_obs", node_obs.shape, "edge_attr", edge_attr.shape)
        # Flatten node_obs
        x = node_obs.view(-1, node_obs.size(-1))
        # Create batch index
        batch = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
        # batch = Batch.from_data_list([Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr) 
        # 						for i in range(node_obs.size(0))])
        # 将 agent_id 直接传递给 SimplifiedAttentionNet
        x = self.gnn(data, agent_id)

        if self.gnn.graph_aggr == "node":
            return x.view(batch_size, -1)
        return x