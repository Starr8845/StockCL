import torch
import torch.nn as nn

# ALSTM model和memory network要分开


class MemoryModule(nn.Module):
    def __init__(self, hidden_size=128, prompt_num = 256):
        super().__init__()
        self.hid_size = hidden_size
        self.memory = nn.Parameter(torch.FloatTensor(size=(prompt_num, self.hid_size)))  
        nn.init.xavier_normal_(self.memory, gain=nn.init.calculate_gain("relu"))

    def memory_enhance(self, repres, get_attention=False):
        attention = torch.einsum("nd,fd->nf", repres, self.memory)
        m = nn.Softmax(dim=1)
        attention = m(attention)
        output = torch.einsum("nf,fd->nd", attention, self.memory)
        # 对应loss constraints的计算
        values, indices = attention.topk(2, dim=1, largest=True, sorted=True)
        largest = self.memory[indices[:,0].squeeze()]
        second_largest = self.memory[indices[:,1].squeeze()]
        distance1 = torch.linalg.vector_norm(repres-largest, dim=1, ord=2)/(self.hid_size)
        distance2 = torch.linalg.vector_norm(repres-second_largest, dim=1, ord=2)/(self.hid_size)
        loss_constraint = torch.mean(distance1)
        temp = (distance1-distance2+1e-3)<0
        if torch.sum(temp)>0:
            loss2 = torch.mean((distance1-distance2+1e-3)[temp])
            loss_constraint+=loss2
        loss_constraint += torch.linalg.matrix_norm(self.memory)
        repres = torch.cat((output, repres), dim=1) # [batch, 4*hidden_size]
        same_proto_mask = ((indices[:,0].reshape(-1,1) - indices[:,0].reshape(1,-1))==0)
        if get_attention:
            return repres, loss_constraint, same_proto_mask, attention
        return repres, loss_constraint, same_proto_mask



class ALSTMModel(nn.Module):  # 实现的是attention LSTM 不是self attention
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)

    def get_repre(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        repres = torch.cat((rnn_out[:, -1, :], out_att), dim=1) # [batch, seq_len, num_directions * hidden_size]
        return repres
    
    def predict(self, repres):
        out = self.fc_out(repres)
        return out[..., 0]
    
    def forward(self, inputs):
        repres = self.get_repre(inputs)
        pred = self.predict(repres)
        return pred

    
class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat
    
    def get_repre(self, x):
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return out[:, -1, :]

    def predict(self, repres):
        return self.fc_out(repres).squeeze()

    def forward(self, inputs):
        repres = self.get_repre(inputs)
        pred = self.predict(repres)
        return pred

    # def forward(self, x):
    #     # x: [N, F*T]
    #     x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    #     x = x.permute(0, 2, 1)  # [N, T, F]
    #     out, _ = self.rnn(x)
    #     return self.fc_out(out[:, -1, :]).squeeze()


from qlib.contrib.model.tcn import TemporalConvNet
class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def get_repre(self, x):
        x = x.reshape(x.shape[0], self.num_input, -1)
        output = self.tcn(x)
        output = output[:, :, -1]
        return output
    
    def predict(self, repres):
        output = self.linear(repres).squeeze()
        return output
    
    def forward(self, inputs):
        repres = self.get_repre(inputs)
        pred = self.predict(repres)
        return pred


from qlib.contrib.model.pytorch_localformer import PositionalEncoding, LocalformerEncoder
class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def get_repre(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        repres = output.transpose(1, 0)[:, -1, :]
        return repres

    def predict(self, repres):
        output = self.decoder_layer(repres)  # [512, 1]
        return output.squeeze()

    def forward(self, inputs):
        repres = self.get_repre(inputs)
        pred = self.predict(repres)
        return pred
    

class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def get_repre(self, x):
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        # hidden = self.fc(hidden)
        # hidden = self.leaky_relu(hidden)
        return hidden

    def predict(self, repres):
        return self.fc_out(repres).squeeze()

    def forward(self, x):
        repres = self.get_repre(x)
        pred = self.predict(repres)
        return pred


class HyperPredictor(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        # hyper network 
        self.hyper_w = nn.Linear(self.hidden_size, self.hidden_size)  #生成的是参数
        self.hyper_b = nn.Linear(self.hidden_size*2,1)
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, repres, repres_memory):
        # repres_memory: [N, 256] # 这里repres应该加上prompts拼接
        # prompt_index: [N,1] #每个样本对应哪个prompt
        # repres: [N, 128]
        prompts = repres_memory[:,:self.hidden_size]
        w = self.hyper_w(prompts) # N*128    # 这里的输入如果改一下呢？ 改成prompt
        b = self.hyper_b(repres_memory) # N*1
        w = self.tanh(w)
        b = self.tanh(b)
        predicted = w*repres 
        predicted = torch.sum(predicted, dim=1, keepdim=True) # N*1
        predicted = predicted+b # N*1
        out = self.fc_out(repres).squeeze()  # (残差连接)
        return (predicted.squeeze() + out)/2 


if __name__=="__main__":
    my_hyper_model = HyperPredictor(hidden_size=128)
    repres_memory = torch.randn(200,256)
    repres = torch.randn(200,128)
    predicted = my_hyper_model(repres, repres_memory)
    print(predicted.shape)
