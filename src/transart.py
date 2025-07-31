"""
Transart

@aandreu7
"""

import sys, torch
import io_transart
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = './imgs/results'
os.makedirs(output_dir, exist_ok=True)

def start_pipeline(content_path: str, style_path: str):
    content_img, style_img = io_transart.load_imgs(content_path, style_path)
    
    generated_img = io_transart.init_g_image(content_img)
    
    model, intermediate_layers = io_transart.load_model()
    
    intermediate_outputs = io_transart.cnn_intermediate_outputs(model, intermediate_layers)
    intermediate_outputs.to(device)
        
    a_C, a_S = io_transart.obtain_activations(intermediate_outputs, content_img, style_img)

    io_transart.optimize_generated_image(generated_img, a_C, a_S, intermediate_outputs, intermediate_layers)

if __name__=="__main__":
    if len(sys.argv) == 1:
        """
        content_path = sys.argv[1]
        style_path = sys.argv[2]
        """
        content_path = "./imgs/van_gogh/content.jpg"
        style_path = "./imgs/van_gogh/style.jpeg"
        
        print(f"Using: {device}")

        start_pipeline(content_path, style_path)
    else:
        raise ValueError("Correct use: transart [content path] [style path]")
    