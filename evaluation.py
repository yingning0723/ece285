from fid_evaluation import FIDWrapper
from diffusion_model import DiffusionModel

device = 'cuda'
model = DiffusionModel(device=device, dataset_name='cifar10', checkpoint_name='cifar10_checkpoint_50.pth')

fid = FIDWrapper(model, batch_size=64, num_samples=5000, device=device)

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = transforms.Compose([transforms.ToTensor(), lambda x: 2 * (x - 0.5)])
dataset = CIFAR10("./datasets", train=False, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

fid.load_dataset_stats(dataloader)

context = model.get_custom_context(5000, 10, device)  # for conditional
score = fid.compute_fid(context=context, timesteps=500, beta1=1e-4, beta2=0.02, schedule='linear')
print("FID Score:", score)