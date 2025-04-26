from torch.nn import init
import torch.nn as nn
import torch

from typing import Tuple
import pandas as pd
import numpy as np
import time


class SumPoolingModel(nn.Module):
	def __init__(
			self, 
			vocab_size,
			input_dim, 
			model, 
			num_layers, 
			num_neurons, 
			device
		):
		"""Create a model based on the request"""
		super(SumPoolingModel, self).__init__()
		self.num_layers = num_layers
		self.num_neurons = num_neurons
		self.vocab_size = vocab_size
		self.input_dim = int(input_dim)
		self.input_dim_mod = self.input_dim
		self.device = device
		self.model_name = model
		# Define the loss function
		self.loss_func = nn.L1Loss()
		# # Embedding Layer
		# self.emb = nn.Embedding(self.vocab_size, self.input_dim)
		# init.uniform_(self.emb.weight,a=-0.5,b=0.5)
		# # self.emb.weight.requires_grad = False		# if it is non trainable
		# self.emb.weight.requires_grad = True
		# Create the model here based on input
		self.model = nn.Linear(self.input_dim_mod, 30)
		self.model_activation = nn.Tanh()
		self.model_out_shape = 30
		init.xavier_uniform_(self.model.weight)
		self.model.bias.data.fill_(0)	
		# Multiple Hidden Layers based on input
		# Neurons in Hidden Layer based on input
		self.rho_mlp_linear = []
		for i in range(num_layers):
			if i == 0:
				self.rho_mlp_linear.append(
					nn.Linear(self.model_out_shape, num_neurons)
				)
			else:
				self.rho_mlp_linear.append(
					nn.Linear(num_neurons, num_neurons)
				)
			init.xavier_uniform_(self.rho_mlp_linear[-1].weight)
			self.rho_mlp_linear[-1].bias.data.fill_(0)
			self.rho_mlp_linear.append(nn.Tanh())
		if self.num_layers == 0:
			self.final_layer = nn.Linear(self.model_out_shape, 1)
		else:	
			for layer_num in range(len(self.rho_mlp_linear)):
				self.add_module(
					"hidden_"+str(layer_num),self.rho_mlp_linear[layer_num]
				)
			self.final_layer = nn.Linear(self.num_neurons, 1)
		init.xavier_uniform_(self.final_layer.weight)
		self.final_layer.bias.data.fill_(0)

	def forward(self, input_tensor):
		"""Lookup the tensor and then continue with feedforward"""
		# embedding: h(x)
		# emb_output = self.emb(input_tensor)
		# emb_shape = emb_output.shape
		# emb_output = emb_output.view(emb_shape[0], emb_shape[1], -1)
		emb_output = input_tensor
		# f(h)
		if self.model_activation is not None:
			model_out = self.model(emb_output)
			model_out = self.model_activation(model_out)
		else:
			model_out, _ = self.model(emb_output)
			model_out = model_out[:, -1, :]  # Just the final state
		# sum(f)
		rho_out = torch.sum(model_out, dim=1).to(self.device)
		# rho(sum(f))
		for layer_num in range(len(self.rho_mlp_linear)):
			rho_out = getattr(self,"hidden_"+str(layer_num))(rho_out)
		final_output = self.final_layer(rho_out)
		return final_output	

	def loss(self, input_tensor, output_tensor):
		"""Loss Computations"""
		predicted_output = self.forward(input_tensor)
		return self.loss_func(predicted_output, output_tensor)


class JPModel(nn.Module):

	def __init__(
			self, 
			vocab_size,
			input_dim, 
			model, 
			num_layers, 
			num_neurons, 
			janossy_k, 
			device
		):
		"""Create a model based on the request"""
		super(JPModel, self).__init__()
		self.num_layers = num_layers
		self.num_neurons = num_neurons
		self.vocab_size = vocab_size
		self.input_dim = int(input_dim/janossy_k)
		self.input_dim_mod = self.input_dim * janossy_k	
		self.device = device
		self.model_name = model
		# Define the loss function
		self.loss_func = nn.L1Loss()
		# # Embedding Layer 
		# self.emb = nn.Embedding(self.vocab_size, self.input_dim)
		# init.uniform_(self.emb.weight,a=-0.5,b=0.5)
		# # self.emb.weight.requires_grad = False		# if it is non trainable
		# self.emb.weight.requires_grad = True
		# Create the model here based on input
		if self.model_name == 'lstm':
			self.model = nn.LSTM(self.input_dim_mod, 50, batch_first=True)
			self.model_activation = None
			self.model_out_shape = 50
		elif self.model_name == 'gru':
			self.model = nn.GRU(self.input_dim_mod, 80, batch_first=True)
			self.model_activation = None
			self.model_out_shape = 80
		else:
			self.model = nn.Linear(self.input_dim_mod, 30)
			self.model_activation = nn.Tanh()
			self.model_out_shape = 30
			init.xavier_uniform_(self.model.weight)
			self.model.bias.data.fill_(0)	
		# Multiple Hidden Layers based on input
		# Neurons in Hidden Layer based on input
		self.rho_mlp_linear = []
		for i in range(num_layers):
			if i == 0:
				self.rho_mlp_linear.append(
					nn.Linear(self.model_out_shape, num_neurons)
				)
			else:
				self.rho_mlp_linear.append(
					nn.Linear(num_neurons, num_neurons)
				)
			init.xavier_uniform_(self.rho_mlp_linear[-1].weight)
			self.rho_mlp_linear[-1].bias.data.fill_(0)
			self.rho_mlp_linear.append(nn.Tanh())
		if self.num_layers == 0:
			self.final_layer = nn.Linear(self.model_out_shape, 1)
		else:	
			for layer_num in range(len(self.rho_mlp_linear)):
				self.add_module(
					"hidden_"+str(layer_num),self.rho_mlp_linear[layer_num]
				)
			self.final_layer = nn.Linear(self.num_neurons, 1)
		init.xavier_uniform_(self.final_layer.weight)
		self.final_layer.bias.data.fill_(0)

	def forward(self, input_tensor):
		"""Lookup the tensor and then continue with feedforward"""
		# Input as a long tensor
		# emb_output = self.emb(input_tensor)
		# emb_shape = emb_output.shape
		# emb_output = emb_output.view(emb_shape[0], emb_shape[1], -1)
		emb_output = input_tensor
		# Feed the obtained embedding to the Janossy Layer
		if self.model_activation is not None:
			model_out = self.model(emb_output)
			model_out = self.model_activation(model_out)
		else:
			model_out, _ = self.model(emb_output)
			model_out = model_out[:, -1, :]  # Just the final state
		if self.model_name in ['lstm','gru']:
			rho_out = model_out
		else:
			summer_out = torch.sum(model_out, dim=1).to(self.device)
			rho_out = summer_out
		for layer_num in range(len(self.rho_mlp_linear)):
			rho_out = getattr(self,"hidden_"+str(layer_num))(rho_out)
		final_output = self.final_layer(rho_out)
		return final_output	

	def loss(self, input_tensor, output_tensor):
		"""Loss Computations"""
		predicted_output = self.forward(input_tensor)
		return self.loss_func(predicted_output, output_tensor)


def build_model(
		input_dim: int, 
		model: str, 
		num_layers: int, 
		num_neurons: int, 
		learning_rate: float,
		iteration: int,
		batch_size: int
	) -> nn.Module:
	janossy_model = JPModel(
		vocab_size = None, 
		input_dim = input_dim,
		model = model,
		num_layers = num_layers,
		num_neurons = num_neurons,
		janossy_k = 1,
		device = "cpu"
	)
	# use Adam Optimizer on all parameters with requires grad as true
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, janossy_model.parameters()), 
		lr = learning_rate
	)
	# checkpoint
	checkpoint_file_name = "{}_{}_{}_{}_{}_checkpoint.pth.tar".format(
		model, num_layers, iteration, learning_rate, batch_size
	)
	return janossy_model, optimizer, checkpoint_file_name

def permute(x):
	return np.random.permutation(x)

def prepare_data(all_data: pd.DataFrame) -> Tuple[torch.Tensor, dict]:
	fname_to_pcomp = {
		"curl": -2.371848,
		"eat-memory":  6.276478,
		"env": -3.252626,
		"figlet": -3.177257,
		"nmap":  5.764093,
		"shasum": -3.238840
	}
	rates = all_data.loc[
		:, all_data.columns.str.startswith("rate_function_")
	].copy(deep = True)
	targets = all_data.loc[
		:, all_data.columns.str.endswith("_usage_node")
	].copy(deep = True)
	node_types = all_data["node_type"].copy(deep = True)
	# identify rows with all-zero input rates
	ncols = rates.shape[1]
	to_drop = rates[rates.le(0).sum(axis = 1).eq(ncols)].index
	# drop rows with all-zero input rates
	rates.drop(to_drop, inplace = True)
	targets.drop(to_drop, inplace = True)
	node_types.drop(to_drop, inplace = True)
	# build X/Y tensors
	X_list = []
	Y_dict = {key: [] for key in targets}
	for idx in range(len(rates)):
		rate = rates.iloc[idx]
		target = targets.iloc[idx]
		node_type = node_types.iloc[idx]
		# x
		xtab = []
		for key, r in rate.items():
			if r > 0:
				fname = key.split("_")[-1]
				xtab.append([fname_to_pcomp[fname], r, int(node_type)])
		X_list.append(xtab)
		# y
		for key, t in target.items():
			Y_dict[key].append([t])
	return X_list, Y_dict

def unison_shuffled(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def to_tensor(l: list) -> torch.Tensor:
	return torch.Tensor(np.array(l))

def train(
		janossy_model: nn.Module,
		optimizer,
		checkpoint_file_name: str,
		batch_size: int,
		num_batches: int,
		num_epochs: int
	):
	device = "cpu"
	# Train over multiple epochs
	start_time = time.time()
	best_val_accuracy = 0.0
	for epoch in range(num_epochs):
		# Do seed and random shuffle of the input
		X, output_X = unison_shuffled(X, output_X)
		#Performing pi-SGD for RNN's
		if model in ['lstm','gru']:
			X = np.apply_along_axis(permute, 1, X)
		for batch in range(num_batches):
			batch_seq = torch.LongTensor(X[batch_size * batch:batch_size * batch + batch_size]).to(device)
			optimizer.zero_grad()
			loss = janossy_model.loss(batch_seq, torch.FloatTensor(output_X[np.array(range(batch_size * batch, batch_size * batch + batch_size))]).to(device))
			loss.backward()
			optimizer.step()
		with torch.no_grad():
			val_output = np.round(janossy_model.forward(torch.LongTensor(V).to(device)).data.cpu().numpy())
			val_loss = janossy_model.loss(torch.LongTensor(V).to(device),torch.FloatTensor(output_V).to(device))
			val_correct = 0
			for j in range(len(output_V)):
				if output_V[j,0] == val_output[j,0]:
					val_correct+=1
			val_accuracy = (1.0*val_correct)/len(output_V)
			if val_accuracy >= best_val_accuracy:
				best_val_accuracy = val_accuracy
				#Save Weights
				torch.save(janossy_model.state_dict(),checkpoint_file_name)	
		print(epoch, loss.data[0],val_loss.data[0])
	end_time = time.time()
	total_training_time = end_time - start_time
	print("Total Training Time: ", total_training_time)


all_data = pd.read_csv("output/output-energy/all_results_no_outliers_TRAIN.csv")
X_train_list, Y_train_dict = prepare_data(all_data)


idxs = {i: [] for i in range(1,7)}
for i, val in enumerate(X_train_list):
	idxs[len(val)].append(i)


xt1 = to_tensor(X_train_list[idxs[1][0]])
yt1 = to_tensor(Y_train_dict["cpu_usage_node"][idxs[1][0]])
xt2 = to_tensor(X_train_list[idxs[2][0]])
yt2 = to_tensor(Y_train_dict["cpu_usage_node"][idxs[2][0]])
xt3 = to_tensor(X_train_list[idxs[3][0]])
yt3 = to_tensor(Y_train_dict["cpu_usage_node"][idxs[3][0]])

xts1, yts2 = unison_shuffled(xt1, yt1)

# SPmodel = SumPoolingModel(6, 3, "linear", 2, 30, "cpu")
# SPmodel.loss(xt, yt)


model = JPModel(6, 3, "lstm", 3, 30, 1, "cpu")
model.loss(xt3,yt3)


# vocab_size = 180
# input_dim = 10
# data = pd.read_csv("output/combined_dataframe.csv")
# emb = nn.Embedding(vocab_size, input_dim)

# data_tensor = torch.LongTensor([
# 	list(data.iloc[i]) for i in range(len(data))
# ])

# res = emb(data_tensor)
# res.size()



