import torch, torchvision, PIL, numpy as np, matplotlib.pyplot as plt
from transart import device, output_dir
import os

class cnn_intermediate_outputs(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model.features # Only takes convolutional part, ignores last part
        self.layers = [str(i) for i, _ in layers] # List of indices we want to extract
        self.eval()
    
    def forward(self, x):
        outputs = []
        x = x.to(next(self.model.parameters()).device)
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                outputs.append(x)
        return outputs
    
def load_imgs(path_content: str, path_style: str) -> (torch.Tensor, torch.Tensor):
    try:
        content_img = (torchvision.io.read_image(path_content).float()/255.0).to(device)
        style_img = (torchvision.io.read_image(path_style).float()/255.0).to(device)
        _, h, w = content_img.shape
        style_img = torchvision.transforms.functional.resize(style_img, [h, w])
        return content_img, style_img
    except:
        raise ValueError(f"Could not read images.")
        
def init_g_image(content_img: torch.Tensor) -> torch.nn.Parameter:
    noise = torch.rand_like(content_img).to(device)
    alpha = 0.2
    noisy_img = content_img + alpha * noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    return torch.nn.Parameter(noisy_img.to(device))

def load_model(model_name: str="vgg19") -> (torch.nn.Module, list):
    if model_name=="vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        """
        Style layers selected:
        
        (1, 0.2),
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU(inplace=True)
        (6, 0.2),
                (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (6): ReLU(inplace=True)
        (11, 0.2),
                (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (11): ReLU(inplace=True)
        (20, 0.2),
                (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (20): ReLU(inplace=True)
        (29, 0.2),
                (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (29): ReLU(inplace=True)
                
        Content layer selected:
    
        (35, 1),
                (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (35): ReLU(inplace=True)
        """
            
        style_layers = [
            (1, 0.2),
            (6, 0.2),
            (11, 0.2),
            (20, 0.2),
            (29, 0.2),
            ]
        
        content_layer = [
            (35, 1),
            ]
    else:
        raise ValueError(f"Model unknown: {model_name}")
        
    model.eval()
    model.to(device)
    
    return model, style_layers+content_layer

def tensor_to_image(tensor: torch.Tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor.mul(255).clamp(0, 255).byte().numpy()
    if tensor.ndim == 4:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    if tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
    return PIL.Image.fromarray(tensor)

def obtain_activations(intermediate_outputs: cnn_intermediate_outputs, 
                       content_img: torch.Tensor, style_img: torch.Tensor) -> (list, list):
    with torch.no_grad():
        activations_content = intermediate_outputs(content_img)
        a_C = [activations_content[-1]]
        
        activations_style = intermediate_outputs(style_img)
        a_S = activations_style[:-1]
        
    return a_C, a_S
    
def compute_gram_matrix(a_l):
    return torch.matmul(a_l, a_l.t())

def compute_content_cost(a_C: list, a_G: list):
    C, H, W = a_G[-1].shape
    
    a_C_ut = a_C[0].permute(1, 2, 0).reshape(H * W, C)
    a_G_ut = a_G[-1].permute(1, 2, 0).reshape(H * W, C)

    J_content = torch.sum((a_C_ut - a_G_ut) ** 2) / (4.0 * C * H * W)

    return J_content

def compute_local_style_cost(a_S_l, a_G_l):
    C, H, W =  a_G_l.shape
    
    a_S_l = a_S_l.reshape(C, H*W)
    a_G_l = a_G_l.reshape(C, H*W)
    
    GMat_S = compute_gram_matrix(a_S_l)
    GMat_G = compute_gram_matrix(a_G_l)
    
    J_style_local = torch.sum((GMat_S - GMat_G) ** 2) / (4.0 * (C * H * W) ** 2)
    
    return J_style_local
    
def compute_global_style_cost(a_S: list, a_G: list, selected_layers: list):
    J_style_global = 0
    
    for i, weight in zip(range(len(a_S)), selected_layers):  
        J_style_local = compute_local_style_cost(a_S[i], a_G[i])
        J_style_global += weight[1] * J_style_local
    
    return J_style_global

def compute_total_cost(J_content, J_style, alpha=10, beta=40):
    return alpha*J_content+beta*J_style

def train_step(generated_image, optimizer, a_C: list, a_S: list, intermediate_outputs, intermediate_layers):
    optimizer.zero_grad()

    a_G = intermediate_outputs(generated_image)

    J_content = compute_content_cost(a_C, a_G)
    J_style = compute_global_style_cost(a_S, a_G[:-1], intermediate_layers)
    J = compute_total_cost(J_content, J_style, alpha=10, beta=500)

    J.backward()
    optimizer.step()

    with torch.no_grad():
        generated_image.clamp_(0.0, 1.0)

    return J.item()

def optimize_generated_image(generated_img, a_C, a_S, intermediate_outputs, intermediate_layers):   
    epochs = 2501
    optimizer = torch.optim.Adam([generated_img], lr=0.01)
    for i in range(epochs):
        train_step(generated_img, optimizer, a_C, a_S, intermediate_outputs, intermediate_layers)
        if i % 250 == 0:
            print(f"Epoch {i} ")
            image = tensor_to_image(generated_img)
            image.save(os.path.join(output_dir, f"Epoch_{i}.png"))