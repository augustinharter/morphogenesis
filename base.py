#%%
import torch
import numpy as np
import cv2
import random
from tqdm import trange

#%%
sobel_x = torch.tensor(16*[16*[[[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]]]], dtype=torch.float32)
sobel_y = sobel_x.transpose(2,3)

targets = dict()
img = cv2.resize(cv2.imread("lizard.png", cv2.IMREAD_UNCHANGED), (64, 64))
mask = img[:,:,3] == 0
img[mask] = [255, 255, 255, 255]
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img = np.array((img[:,:,0], img[:,:,1], img[:,:,2]))
#print(img.shape)
targets["lizard"] = img

#%%
class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.dense1 = torch.nn.Linear(48, 128)
    self.dense2 = torch.nn.Linear(128, 16)

  def forward(self, input):
    return self.dense2(torch.relu(self.dense1(input)))

#%%
def perceive(state_grid):
  global sobel_x, sobel_y
  # Convolve sobel filters with states
  # in x, y and channel dimension.
  compute_grid = state_grid[None,:,:,:]
  grad_x = torch.conv2d(compute_grid, sobel_x, padding=1, groups=1)[0]
  grad_y = torch.conv2d(compute_grid, sobel_y, padding=1, groups=1)[0]
  # Concatenate the cell’s state channels,
  # the gradients of channels in x and 
  # the gradient of channels in y.
  #print(compute_grid.shape, grad_x.shape, grad_y.shape)
  perception_grid = torch.cat((state_grid, grad_x, grad_y), dim=0)
  #print(perception_grid.shape)
  return perception_grid

def update(perception_vector, net):
  # The following pseudocode operates on
  # a single cell’s perception vector. 
  # Our reference implementation uses 1D
  # convolutions for performance reasons.
  return net(perception_vector)

def stochastic_update():
  # Zero out a random fraction of the updates.  
  return torch.tensor(torch.rand(64, 64) < 0.5)

def alive_masking(state_grid):
  # Take the alpha channel as the measure of “life”.
  alive = torch.tensor(torch.max_pool2d(state_grid[None, 3, :, :], (3,3), padding=1, stride=1) > 0.1, dtype=torch.float)
  state_grid = state_grid * alive
  return state_grid

# PREPERATION
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

def train(sample_grid, target):
  net.zero_grad()
  for step in range(64):
    ds_grid = torch.zeros_like(sample_grid)
    perception_grid = perceive(sample_grid)
    stochastic_mask = stochastic_update()
    for j in range(64):
      for k in range(64):
        if stochastic_mask[j,k]==1 and sample_grid[3,j,k]!=0:
          ds_grid[:,j,k] = net(perception_grid[:,j,k])
    sample_grid = sample_grid + ds_grid
    sample_grid = alive_masking(sample_grid)

  loss = criterion(sample_grid[:3,:,:], target)
  #print("before gradient")
  loss.backward()
  #print("after gradient")
  optimizer.step()
  return sample_grid
    
seed = torch.zeros(16, 64, 64, dtype=torch.float32)
seed[3:, 64//2, 64//2] = 1.0

def pool_training(target, iterations):
  global targets, seed
  # Set alpha and hidden channels to (1.0).
  target = torch.tensor(targets[target], dtype=torch.float32)
  pool = [seed] * 1024
  for i in range(iterations):
    print(f"iteration {i+1}")
    idxs, batch = zip(*random.sample(list(enumerate(pool)), 32))
    # Sort by loss, descending.
    batch = sorted(batch, key=lambda x: float(criterion(x[:3,:,:], target)), reverse=True)
    # Replace the highest-loss sample with the seed.
    batch[0] = seed
    # Perform training.
    for j in trange(32):
      # Place outputs back in the pool.
      pool[idxs[j]] = train(batch[j], target)

def run(seed, visual=False, load=False):
  if load:
    net.load_state_dict(torch.load("local_rules_net.pt"))
  net.eval()
  with torch.no_grad():
    for step in range(100):
      ds_grid = torch.zeros_like(seed)
      perception_grid = perceive(seed)
      stochastic_mask = stochastic_update()
      for j in range(64):
        for k in range(64):
          if stochastic_mask[j,k]==1 and seed[3,j,k]!=0:
            ds_grid[:,j,k] = net(perception_grid[:,j,k])
      seed = seed + ds_grid
      seed = alive_masking(seed)
      if visual:
        img = seed.numpy()
        img = np.array((img[0,:,:], img[1,:,:], img[2,:,:]))
        cv2.imshow("cellgrid", img)
        cv2.waitKey()
  if not visual:
    seed = seed.numpy()
    print(np.array((seed[0,:,:], seed[1,:,:], seed[2,:,:])))
    

  
#%%
# TRAINING
pool_training('lizard', 1)

torch.save(net.state_dict(), "local_rules_net.pt")
run(seed, load=True, visual=True)
# %%