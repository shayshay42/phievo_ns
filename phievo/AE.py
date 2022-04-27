import torch
import torchvision
#import keras
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(df): #pass a pandas frame of the 1000 sequences of length 250 from the output
  sequences = df.astype(np.float32).to_numpy().transpose().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  print("n_seq, seq_len, n_features", n_seq, seq_len, n_features)
  return dataset, seq_len, n_features


#LSTM AUTOENCODER

class Enc(nn.Module):
  def __init__(self):
    super(Enc, self).__init__()
    self.rnn1 = nn.LSTM(
      input_size=1,
      hidden_size=4,
      num_layers=1,
      batch_fist=False
      )   
    self.fc1 = nn.Linear(in_features=1000, out_features=512)
    self.fc2 = nn.Linear(in_features= 512, out_features=250)
  def forward(self, x):
    x = x.reshape((-1, 250, 1))
    y, (hidden_n, cell_n) = self.rnn1(x)
    stacked = cell_n.reshape((1000,1))
    cn_hid = torch.relu(self.fc1(stacked))
    cn = self.fc2(cn_hid)
    return y,cn

class Dec(nn.Module):
  def __init__(self):
    super(Dec, self).__init__()
    self.rnn1 = nn.LSTM(
      input_size=4,
      hidden_size=1,
      num_layers=1,
      batch_first=False
    )
  def forward(self, x):
    inp,c0 = x
    h0 = torch.randn(1,250,1)
    yrec, (hidden_n, cell_n) = self.rnn1(inp, (h0,c0))
    return yrec

class lstm_simp(nn.Module):
  def __init__(self):
    super(lstm_simp, self).__init__()
    self.encoder = Enc().to(device)
    self.decoder = Dec().to(device)
  def forward(self, x):
    y,cn = self.encoder(x)
    yrec = self.decoder([y,cn])
    return yrec


#GRU AUTOENCODER

class Encg(nn.Module):
  def __init__(self):
    super(Encg, self).__init__()
    self.rnn1 = nn.GRU(
      input_size=1,
      hidden_size=4,
      num_layers=1,
      batch_first=False
      )
    self.fc1 = nn.Linear(in_features=1000, out_features=512)
    self.fc2 = nn.Linear(in_features= 512, out_features=250)
  def forward(self, x):
    x = x.reshape((-1, 250, 1))
    y, hidden_n = self.rnn1(x)
    stacked = hidden_n.reshape((1,1000)) #not sure if this is necessary actually cuz i get 1 idk
    hn_hid = torch.relu(self.fc1(stacked))
    hn = self.fc2(hn_hid)
    return y,hn

class Decg(nn.Module):
  def __init__(self):
    super(Decg, self).__init__()
    self.rnn1 = nn.GRU(
      input_size=4,
      hidden_size=1,
      num_layers=1,
      batch_first=False
    )
  def forward(self, inp,h0):
    yrec, hidden_n = self.rnn1(inp, h0.reshape(1,250,1))
    return yrec

class gru_simp(nn.Module):
  def __init__(self):
    super(gru_simp, self).__init__()
    self.encoder = Encg().to(device)
    self.decoder = Decg().to(device)
  def forward(self, x):
    y,hn = self.encoder(x)
    yrec = self.decoder(y,hn)
    return yrec.reshape(250,1)



def train_recurAE(model, train_dataset, val_dataset, n_epochs):
  print("now starting training!", len(train_dataset))
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in tqdm(range(1, n_epochs + 1)):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      #print("two and a half")
      optimizer.zero_grad()
      #print("2.6")
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      #print("2.75")
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      optimizer.step()
      #print("2.8")
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history












### DRAFTS ###


















class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


####

class linearEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(linearEncoder, self).__init__()
        inner, inner2,inner3,inner4,inner5 = 512,256,128,32,16
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=inner)
        self.encoder_layer2 = nn.Linear(in_features=inner, out_features=inner2)
        self.encoder_layer3 = nn.Linear(in_features=inner2, out_features=inner3)
        self.encoder_layer4 = nn.Linear(in_features=inner3, out_features=inner4)
        self.encoder_output_layer = nn.Linear(in_features=inner4, out_features=inner5)
        # self.encoder_layer6 = nn.Linear(in_features=inner5, out_features=inner6)
        # self.encoder_output_layer = nn.Linear(in_features=inner6, out_features=inner7)
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.encoder_layer2(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer3(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer4(activation)
        # activation = torch.relu(activation)
        # activation = self.encoder_layer5(activation)
        # activation = torch.relu(activation)
        # activation = self.encoder_layer6(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        return code

class linearDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(linearDecoder, self).__init__()
        inner, inner2,inner3,inner4,inner5 = 512,256,128,32,16#2500,1250,512,256,128,32,16
        # self.decoder_hidden_layer = nn.Linear(in_features=inner7, out_features=inner6)
        # self.dencoder_layer2 = nn.Linear(in_features=inner6, out_features=inner5)
        self.decoder_hidden_layer = nn.Linear(in_features=inner5, out_features=inner4)
        self.dencoder_layer4 = nn.Linear(in_features=inner4, out_features=inner3)
        self.dencoder_layer5 = nn.Linear(in_features=inner3, out_features=inner2)
        self.dencoder_layer6 = nn.Linear(in_features=inner2, out_features=inner)
        self.decoder_output_layer = nn.Linear(in_features=inner, out_features=kwargs["input_shape"])
    def forward(self, features):
        code1 = torch.relu(features)
        activation = self.decoder_hidden_layer(code1)
        activation = torch.relu(activation)
        # activation = self.dencoder_layer2(activation)
        # activation = torch.relu(activation)
        # activation = self.dencoder_layer3(activation)
        # activation = torch.relu(activation)
        activation = self.dencoder_layer4(activation)
        activation = torch.relu(activation)
        activation = self.dencoder_layer5(activation)
        activation = torch.relu(activation)
        activation = self.dencoder_layer6(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

class linearAE(nn.Module):
    def __init__(self, **kwargs):
        super(linearAE, self).__init__()
        self.encoder = linearEncoder(input_shape=kwargs["input_shape"])
        self.decoder = linearDecoder(input_shape=kwargs["input_shape"])
    def forward(self, features):
        latent = self.encoder(features)
        return self.decoder(latent)

def train_linear(model, device, input_len, train_loader, n_epochs=100):
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.MSELoss()
  for epoch in tqdm(range(n_epochs)):
      loss = 0
      for batch_features in train_loader:
          # reshape mini-batch data to [N, 784] matrix
          # load it to the active device
          batch_features = batch_features.view(-1, input_len).to(device)
          # reset the gradients back to zero
          # PyTorch accumulates gradients on subsequent backward passes
          optimizer.zero_grad()
          # compute reconstructions
          outputs = model(batch_features)
          # compute training reconstruction loss
          train_loss = criterion(outputs, batch_features)
          # compute accumulated gradients
          train_loss.backward()
          # perform parameter update based on current gradients
          optimizer.step()
          # add the mini-batch training loss to epoch loss
          loss += train_loss.item()
      # compute the epoch training loss
      loss = loss / len(train_loader)
      # display the epoch training loss
      print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epochs, loss))
  return model, model.eval()

def plot_latent(autoencoder, data, num_batches=100):
  for i, (x, y) in enumerate(data):
      z = autoencoder.encoder(x.to(device))
      z = z.to('cpu').detach().numpy()
      plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
      if i > num_batches:
          plt.colorbar()
          break


class convEnc(nn.Module):
    def __init__(self, **kwargs):
        super(convEnc, self).__init__()
        inner, inner2,inner3,inner4,inner5,inner6,inner7 = 2500,1250,512,256,128,32,16
        self.encoder_hidden_layer = nn.Conv1d(in_features=kwargs["input_shape"], out_features=inner)
        self.encoder_output_layer = nn.Linear(in_features=inner6, out_features=inner7)
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        return code
class convDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(linearDecoder, self).__init__()
        inner, inner2,inner3,inner4,inner5,inner6,inner7 = 2500,1250,512,256,128,32,16
        self.decoder_hidden_layer = nn.Linear(in_features=inner7, out_features=inner6)
        self.dencoder_layer6 = nn.Linear(in_features=inner2, out_features=inner)
        self.decoder_output_layer = nn.Linear(in_features=inner, out_features=kwargs["input_shape"])
    def forward(self, features):
        code1 = torch.relu(features)
        activation = self.decoder_hidden_layer(code1)
        activation = torch.relu(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

class convAE(nn.Module):
    def __init__(self, **kwargs):
        super(linearAE, self).__init__()
        self.encoder = linearEncoder(input_shape=kwargs["input_shape"])
        self.decoder = linearDecoder(input_shape=kwargs["input_shape"])
    def forward(self, features):
        latent = self.encoder(features)
        return self.decoder(latent)

class conv_recur_AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        inner = 2500
        kernel = 1000
        stride = 60
        self.encoder_hidden_layer = nn.Conv1d(in_features=kwargs["input_shape"], out_features=inner)
        self.encoder_output_layer = nn.Linear(in_features=inner6, out_features=inner7)
        #symmetry
        self.decoder_hidden_layer = nn.Linear(in_features=inner7, out_features=inner6)
        self.decoder_output_layer = nn.Linear(in_features=inner, out_features=kwargs["input_shape"])

    def forward(self, features):
        print("these are the feats",features)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code1 = torch.relu(code)
        activation = self.decoder_hidden_layer(code1)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code

class linear_attention_AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        inner = 2500
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=inner)
        self.encoder_output_layer = nn.Linear(in_features=inner6, out_features=inner7)
        #symmetry
        self.decoder_hidden_layer = nn.Linear(in_features=inner7, out_features=inner6)
        self.decoder_output_layer = nn.Linear(in_features=inner, out_features=kwargs["input_shape"])

    def forward(self, features):
        print("these are the feats",features)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code1 = torch.relu(code)
        activation = self.decoder_hidden_layer(code1)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code