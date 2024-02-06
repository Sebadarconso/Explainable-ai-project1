## Imports 
from unittest import result
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import f1_score
from torch.autograd import Variable

## DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

## INTEGRATED GRADIENTS FUNCTIONS

def interpolate_images(baseline, image, alphas):
	alphas_x = alphas[:, None, None, None] # alpha values for x interpolation
	baseline_x = baseline.unsqueeze(0) 
	input_x = image.unsqueeze(0)
	delta = input_x - baseline_x
	images = baseline_x + alphas_x * delta
	return images

# plot interpolated images 
def plot_interpolated_images(interpolate_images, m_steps, step):
	alphas = torch.linspace(0, 1, steps=m_steps + 1)
	
	fig = plt.figure(figsize=(20, 20))
	for i in range(0, interpolate_images.size(1), step):
		plt.subplot(1, len(range(0, interpolate_images.size(1), step)), i // step + 1)
		image_np = interpolate_images[0, i].numpy().transpose(1, 2, 0)

		plt.imshow(image_np)
		plt.title(f"alpha: {alphas[i]:.1f}")
		plt.axis("off")
	plt.tight_layout()
	plt.show()

# compute gradients
def compute_gradients(interpolated_images, model, target_class_idx):
	gradients = []
	model.zero_grad() # before was model.eval()
	images = interpolated_images.clone().detach().requires_grad_(True)
	output = model(images.squeeze(0).to(device))
	probs = F.softmax(output, dim=1)[: , target_class_idx]
	# probs = torch.max(probs)
	gradients = torch.autograd.grad(probs, images, 
								grad_outputs=torch.ones_like(probs),
							 	create_graph=True)[0]
	
	return gradients
	
# Integral approximation
def integral_approximation(gradients):
	grads = (gradients[:-1] + gradients[1:]) / torch.Tensor([2.0])
	# avg_grads = torch.mean(grads, dim=0)
	avg_grads = torch.mean(grads.detach(), axis=0)
	return avg_grads

# Single batch calculation
def one_batch(baseline, image, alpha_batch, target_class_idx, model):

	interpolated_path_input_batch = interpolate_images(baseline=baseline, 
													   image=image, 
													   alphas=alpha_batch)
	
	gradients = compute_gradients(interpolated_images=interpolated_path_input_batch,
								  target_class_idx=target_class_idx,
								  model=model)


	return gradients

## INTEGRATED GRADIENTS
def integrated_gradients(baseline, image, target_class_idx, model, m_steps, batch_size):
	# Generate alphas.
	alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps + 1)

	# Collect gradients.
	gradient_batches = []

	# Iterate alphas range and batch computation
	for alpha in tqdm(range(0, len(alphas), batch_size)):
		from_ = alpha
		to = min(from_ + batch_size, len(alphas))
		alpha_batch = alphas[from_:to]

		gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx, model)
		gradient_batches.append(gradient_batch)

	# Concatenate path gradients together row-wise into a single tensor.
	total_gradients = torch.cat(gradient_batches, dim=0)

	# Integral approximation through averaging gradients.
	avg_gradients = integral_approximation(gradients=total_gradients)

	# Scale integrated gradients with respect to input.
	integrated_gradients = (image - baseline) * avg_gradients

	return integrated_gradients


## PLOT IMAGES 
def plot_image(img_plot, ig_attr_captum, ig_attr_custom, baseline):
	# img_plot = img_plot.numpy().transpose(1, 2, 0)
	img_plot = img_plot.squeeze(0).permute(1, 2, 0)
	captum_plot = ig_attr_captum
	custom_plot = ig_attr_custom

	

	fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), gridspec_kw = {'wspace':0.1, 'hspace':0.2})

	## baseline 
	ax[0, 0].imshow(baseline.squeeze().permute(1, 2, 0))
	ax[0, 0].set_title('Baseline')

	## original image]
	ax[0, 1].imshow(img_plot)
	ax[0, 1].set_title('Original Image')

	## captum
	ax[1, 0].imshow(captum_plot)
	ax[1, 0].set_title('Captum')

	## custom
	ax[1, 1].imshow(custom_plot)
	ax[1, 1].set_title('Custom')

	## overlay captum
	ax[2, 0].imshow(img_plot)
	ax[2, 0].imshow(captum_plot, cmap='inferno', alpha=0.9)
	ax[2, 0].set_title('Overlay Captum')

	## overlay custom
	ax[2, 1].imshow(img_plot)
	ax[2, 1].imshow(custom_plot, cmap='inferno', alpha=0.9)
	ax[2, 1].set_title('Overlay Custom')


	plt.subplots_adjust(hspace=0.2, wspace=0.5)
	plt.tight_layout()



## DEFINE IMAGE CLASSIFICATION MODEL
class ImageClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_layer_1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2))
		self.conv_layer_2 = nn.Sequential(
			nn.Conv2d(64, 512, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2))
		self.conv_layer_3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2)) 
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=512*3*3, out_features=2))
	def forward(self, x: torch.Tensor):
		x = self.conv_layer_1(x)
		x = self.conv_layer_2(x)
		x = self.conv_layer_3(x)
		x = self.conv_layer_3(x)
		x = self.conv_layer_3(x)
		x = self.conv_layer_3(x)
		x = self.classifier(x)
		return x
	
## DEFINE DATASET CLASS
class MuffinChihuahuaData(Dataset):
	def __init__(self,x,y,transform=None):
		self.x = x.reset_index()
		self.y = y.reset_index()
		self.transform = transform        
	def __len__(self):
		return self.x.shape[0]
	def load_image(self,path):
		prefix = "dataset/"
		return Image.open(os.path.join(prefix,path))
	def __getitem__(self,index):
		image = self.load_image(self.x.iloc[index].filename)
		label = self.y.iloc[index].label
		if self.transform:
			image = self.transform(image)
		sample = {"image":image,"label":label}
		return sample
	
## TRAINING AND TESTING FUNCTIONS
def train_step(model:torch.nn.Module, 
			   dataloader:torch.utils.data.DataLoader, 
			   loss_fn:torch.nn.Module, 
			   optimizer:torch.optim.Optimizer):    
	# Put model in train mode
	model.to(device)
	model.train()
	# Setup train loss and train accuracy values
	train_loss,train_acc = 0,0
	# Loop through DataLoader batches
	for batch,sample_batched in tqdm(enumerate(dataloader), total=len(dataloader)):
		# Send data to target device
		X = sample_batched["image"].to(device)
		y = sample_batched["label"].to(device)
		# Forward pass
		y_pred = model(X)
		# Calculate  and accumulate loss
		loss = loss_fn(y_pred,y)
		train_loss += loss.item() 
		# Optimizer zero grad
		optimizer.zero_grad()
		# Loss backward
		loss.backward()
		# Optimizer step
		optimizer.step()
		# Calculate and accumulate accuracy metric across all batches
		y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
		train_acc += (y_pred_class == y).sum().item()/len(y_pred)
	# Adjust metrics to get average loss and accuracy per batch 
	train_loss = train_loss / len(dataloader)
	train_acc = train_acc / len(dataloader)
	return train_loss,train_acc


def val_step(model:torch.nn.Module, 
			  dataloader:torch.utils.data.DataLoader, 
			  loss_fn:torch.nn.Module):    
	# Put model in eval mode
	model.to(device)
	model.eval() 
	# Setup validation loss and validation accuracy values
	val_loss,val_acc, val_f1 = 0,0,0

	## lists to store predictions and labels for f1 score
	all_val_predictions, all_val_targets = [], []

	# Turn on inference context manager
	with torch.inference_mode():
		# Loop through DataLoader batches
		for batch,sample_batched in tqdm(enumerate(dataloader), total=len(dataloader)):
			# Send data to target device
			X = sample_batched["image"].to(device)
			y = sample_batched["label"].to(device)            
			# Forward pass
			val_pred_logits = model(X)
			# Calculate and accumulate loss
			loss = loss_fn(val_pred_logits, y)
			val_loss += loss.item()
			# Calculate and accumulate accuracy
			val_pred_labels = val_pred_logits.argmax(dim=1)
			val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

			# Collect predictions and targets for F1 score calculation
			all_val_predictions.extend(val_pred_labels.cpu().numpy())
			all_val_targets.extend(y.cpu().numpy())

	# Adjust metrics to get average loss and accuracy per batch 
	# Calculate F1 score
	val_f1 = f1_score(all_val_targets, all_val_predictions, average='macro')

	val_loss = val_loss / len(dataloader)
	val_acc = val_acc / len(dataloader)
	return val_loss,val_acc,val_f1


def train(model:torch.nn.Module, 
		  train_dataloader:torch.utils.data.DataLoader, 
		  val_dataloader:torch.utils.data.DataLoader, 
		  optimizer:torch.optim.Optimizer,
		  loss_fn:torch.nn.Module = nn.CrossEntropyLoss(),
		  epochs:int = 5,
		  split:int = 0):
	# Create empty results dictionary
	results = {"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
		"val_f1": []
	}
	# Instantiating the best validation accuracy
	best_val = 0
	# Loop through training and validation steps for a number of epochs
	for epoch in range(epochs):
		print(f"Epoch {epoch+1}/{epochs}")
		train_loss, train_acc = train_step(model=model,
										   dataloader=train_dataloader,
										   loss_fn=loss_fn,
										   optimizer=optimizer)
		val_loss, val_acc, val_f1 = val_step(model=model,
			dataloader=val_dataloader,
			loss_fn=loss_fn) 
		# Saving the model obtaining the best validation accuracy through the epochs
		if val_acc > best_val:
			best_val = val_acc
			checkpoint = {"model": ImageClassifier(),
						  "state_dict": model.state_dict(),
						  "optimizer": optimizer.state_dict()}
			checkpoint_name = "checkpoint/checkpoint_"+str(split)+".pth"
			torch.save(checkpoint, checkpoint_name)    
		# else:
		# 	continue
		# Print out what's happening
		print(
			f"Epoch: {epoch+1} | "
			f"train_loss: {train_loss:.4f} | "
			f"train_acc: {train_acc:.4f} | "
			f"val_loss: {val_loss:.4f} | "
			f"val_acc: {val_acc:.4f} | " 
			f"val_f1: {val_f1:.4f}"
		)
		# Update results dictionary
		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["val_loss"].append(val_loss)
		results["val_acc"].append(val_acc)
		results["val_f1"].append(val_f1)

	# Return the filled results at the end of the epochs
	return results