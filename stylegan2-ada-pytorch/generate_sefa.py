import torch_utils
import pickle
import torch
import argparse
import numpy as np
import PIL.Image
import os

parser = argparse.ArgumentParser(description='Geration using Sefa')
parser.add_argument('--pretrained_network', default='./experiments/00005-dataset-auto1-gamma50-bg/network-snapshot-001216.pkl', help='pretrained network')
parser.add_argument('--eigvec_path', default='factor.pt', help='directory for factor.pt')
parser.add_argument('--degree', default=10, type=int, help='degree of direction')
parser.add_argument('--index', default=0, type=int, help='direction index')
parser.add_argument('--options', default="Female/Old/Glasses", help='Choose options such as "Female/Old/Glasses" to create intentionally.')
parser.add_argument('--truncation_psi', default=0.5, type=float, help='truncation_psi')
parser.add_argument('--noise_mode', default='const', help='noise_mode')
parser.add_argument('--images', default=False, help='generate several image')
parser.add_argument('--save_folder', default='./images', help='Location to save images')
parser.add_argument('--device', default='cpu', help='cpu or cuda')

args = parser.parse_args()
device = torch.device(args.device)

def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1).squeeze(0)
    
    return images

def parse_options(options, eigvec_path):
    opt = options.split('/')
    gender, age, glasses = opt[0], opt[1], opt[2]
    
    if gender == 'Female':
        gender_direction = get_direction(eigvec_path=eigvec_path, index=205, degree=10)
    elif gender == 'Male':
        gender_direction = -get_direction(eigvec_path=eigvec_path, index=205, degree=10)
    
    if age == 'Old':
        age_direction = -get_direction(eigvec_path=eigvec_path, index=294, degree=10)
    elif age == 'Young':
        age_direction = get_direction(eigvec_path=eigvec_path, index=294, degree=10)
        
    if glasses == 'Glasses':
        glasses_direction = get_direction(eigvec_path=eigvec_path, index=175, degree=10)
    elif gender == 'not_wearing_glasses':
        glasses_direction = -get_direction(eigvec_path=eigvec_path, index=175, degree=10)
    
    return gender_direction[0] + age_direction[0] + glasses_direction[0]

def get_direction(eigvec_path=None, index=0, degree=0):
    eigvec = torch.load(eigvec_path)["eigvec"].to(device)
    current_eigvec = eigvec[:, index].unsqueeze(0)
    direction = degree * current_eigvec
    return direction

def generate_images(z, label, truncation_psi, noise_mode, direction):
    img1 = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img2 = G(z + direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img3 = G(z - direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    return torch.cat([img3, img1, img2], 0)

def generate(direction, truncation_psi, noise_mode, images, save_folder):
    with open(args.pretrained_network, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    z = torch.randn([1, G.z_dim]).to(device)    # latent codes
    c = None                                # class labels (not used in this example)
    w = G.mapping(z+direction, c, truncation_psi=0.5, truncation_cutoff=8) 
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    img = postprocess(img)
    
    if images == True:
        img = generate_images(z, None, truncation_psi, noise_mode, direction)
        
    image = PIL.Image.fromarray(img)
    path = 'sample_image.jpg'
    os.makedirs(save_folder, exist_ok=True)
    image.save(os.path.join(save_folder, path))
        
    return img

#----------------------------------------------------------------------------
if __name__ == "__main__":
    if args.options is not None:
        direction = parse_options(options=args.options, eigvec_path=args.eigvec_path)
    else:
        direction = get_direction(eigvec_path=args.eigvec_path, index=args.index, degree=args.degree)
        
    generate(direction=direction, truncation_psi=args.truncation_psi, noise_mode=args.noise_mode,
            images=args.images, save_folder=args.save_folder)
#----------------------------------------------------------------------------