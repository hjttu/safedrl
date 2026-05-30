import torch
import torch.nn as nn

"""LSTM modules."""

import torch
import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        # Replacing GRU with LSTM
        self.lstm = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        # print("x.shape before: ", x.shape)
        # print("hxs.shape before: ", hxs.shape)
        # print("masks.shape before: ", masks.shape)

        # Check if hxs contains both h and c; if not, initialize c as zeros

        h, c = torch.chunk(hxs, chunks=2, dim=2)
        # print("1 h.shape: ", h.shape)
        # print("1 c.shape: ", c.shape)

        
        if x.size(0) == h.size(0):

            # print("h.shape: ", h.shape)
            # print("c.shape: ", c.shape)

            # 确保 masks 正确应用
            h = (h * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous()
            c = (c * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous()

            # print("after mask h.shape: ", h.shape)
            # print("after mask c.shape: ", c.shape)
            # 调用 LSTM
            x, (h, c) = self.lstm(
                x.unsqueeze(0),
                (h, c)
            )

            # 移除批次维度
            x = x.squeeze(0)

            # 确保 h 和 c 的维度正确
            h = h.transpose(0, 1)
            c = c.transpose(0, 1)
        else:
            N = h.size(0)
            T = int(x.size(0) / N)

            # Unflatten
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            has_zeros = [0] + has_zeros + [T]

            h = h.transpose(0, 1)
            c = c.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp_h = h * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1).contiguous()
                temp_c = c * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1).contiguous()
                
                lstm_scores, (h, c) = self.lstm(x[start_idx:end_idx], (temp_h, temp_c))
                outputs.append(lstm_scores)

            x = torch.cat(outputs, dim=0)
            x = x.reshape(T * N, -1)

            h = h.transpose(0, 1)
            c = c.transpose(0, 1)

        x = self.norm(x)
        # hxs = (h, c)

        # print("after h.shape: ", h.shape)
        # print("after c.shape: ", c.shape)
        hxs = torch.cat((h, c), dim=2)

        # print("x.shape: ", x.shape)
        # print("after hxs.shape: ", hxs.shape)

        return x, hxs

