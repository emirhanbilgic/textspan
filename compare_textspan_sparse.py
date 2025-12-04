#!/usr/bin/env python
"""
Generate comparison heatmaps for TextSpan vs Sparse TextSpan (with OMP).
For each image, create comparison showing:
- Original TextSpan heatmaps for all prompts
- Sparse TextSpan heatmaps for all prompts (with OMP denoising)
"""
import numpy as np
import torch
from PIL import Image
import os
import cv2
import einops
from torch.nn import functional as F
from utils.factory import create_model_and_transforms, get_tokenizer
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt
import glob
import re

# Configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pretrained = 'laion2b_s32b_b82k'
model_name = 'ViT-L-14'
data_path = '/Users/emirhan/Desktop/clip_text_span/data'
output_path = '/Users/emirhan/Desktop/clip_text_span/comparison_output'

# Create output directory
os.makedirs(output_path, exist_ok=True)

print(f"Using device: {device}")
print("Loading model...")

# Load model
model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
model.to(device)
model.eval()
tokenizer = get_tokenizer(model_name)

print("Model loaded successfully!")
print(f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

# Hook the PRS logger
prs = hook_prs_logger(model, device)

# Prompts to test
prompts = [
    'a photo of a automobile',
    'a photo of a plane', 
    'a photo of a bird',
    'a photo of a cat',
    'a photo of a dog'
]

# Classes for dictionary building
classes = ['automobile', 'plane', 'bird', 'cat', 'dog']

print(f"\nPrompts: {prompts}")
print(f"Classes: {classes}\n")


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual without training.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], atom rows, L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()  # residual starts as x
    for _ in range(max_atoms):
        # correlations with residual
        c = (r @ D.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        # mask already selected
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        # Solve least squares on selected atoms: s = argmin ||x - s^T D_S||^2
        D_S = D[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        # Regularize G slightly for stability
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t,1]
        x_hat = (s.t() @ D_S).to(x.dtype)  # [1, d]
        r = (x - x_hat)
        # Early stop if residual very small
        if float(torch.norm(r) <= tol):
            break
    # Return normalized residual (fallback to x if degenerate)
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


def build_dictionary_for_class(class_embeddings, target_idx, similarity_threshold=0.9):
    """
    Build dictionary for OMP by including all classes except target class.
    Filter out embeddings with cosine similarity > similarity_threshold.
    
    Args:
        class_embeddings: [num_classes, d] normalized embeddings
        target_idx: index of the target class
        similarity_threshold: filter out if cosine similarity > this value
    
    Returns:
        D: [K, d] dictionary tensor (normalized)
        kept_indices: list of indices kept in dictionary
    """
    target_emb = class_embeddings[target_idx:target_idx+1]  # [1, d]
    
    # Get all embeddings except target
    other_indices = [i for i in range(len(class_embeddings)) if i != target_idx]
    
    if len(other_indices) == 0:
        return target_emb.new_zeros((0, target_emb.shape[-1])), []
    
    other_embs = class_embeddings[other_indices]  # [K-1, d]
    
    # Compute cosine similarities with target
    sims = (other_embs @ target_emb.t()).squeeze(-1).abs()  # [K-1]
    
    # Filter by similarity threshold
    keep_mask = sims < similarity_threshold
    kept_indices = [other_indices[i] for i, keep in enumerate(keep_mask.tolist()) if keep]
    
    if keep_mask.sum() == 0:
        return target_emb.new_zeros((0, target_emb.shape[-1])), []
    
    D = other_embs[keep_mask]  # [K', d]
    D = F.normalize(D, dim=-1)
    
    return D, kept_indices


def compute_heatmap_from_embedding(model, prs, image_tensor, text_embedding, device):
    """
    Compute heatmap for a given text embedding (original or sparse).
    
    Args:
        model: CLIP model
        prs: PRS logger
        image_tensor: preprocessed image tensor
        text_embedding: [1, d] text embedding (normalized)
        device: computation device
    
    Returns:
        heatmap: [H, W] numpy array
    """
    # Run the image through the model
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image_tensor.to(device), 
                                          attn_method='head', 
                                          normalize=False)
        attentions, mlps = prs.finalize(representation)
    
    # Compute attention map with text embedding
    attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ text_embedding.T
    
    # Interpolate to image size
    attention_map = F.interpolate(
        einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
        scale_factor=model.visual.patch_size[0],
        mode='bilinear'
    ).to(device)
    
    attention_map = attention_map[0, 0].detach().cpu().numpy()
    
    return attention_map


def create_overlay(image_pil, heatmap, v_min, v_max, alpha=0.5):
    """
    Create overlay of heatmap on image using provided min/max for normalization.
    
    Args:
        image_pil: PIL Image
        heatmap: [H, W] numpy array
        v_min: global minimum value for normalization
        v_max: global maximum value for normalization
        alpha: overlay alpha
    
    Returns:
        overlay: numpy array (RGB)
    """
    # Normalize to 0-255 using global min/max (RELATIVE mode)
    if v_max - v_min > 0:
        v_normalized = np.clip((heatmap - v_min) / (v_max - v_min), 0, 1)
    else:
        v_normalized = heatmap * 0
    v_uint8 = np.uint8(v_normalized * 255)
    
    # Apply colormap (JET - blue is low, red is high)
    heatmap_colored = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image size
    img_array = np.array(image_pil)
    heatmap_resized = cv2.resize(heatmap_rgb, (img_array.shape[1], img_array.shape[0]))
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_resized, alpha, 0)
    
    return overlay


# Get all image files
image_files = glob.glob(os.path.join(data_path, '*.png')) + \
              glob.glob(os.path.join(data_path, '*.jpg')) + \
              glob.glob(os.path.join(data_path, '*.jpeg'))

print(f"Found {len(image_files)} images")
print(f"Testing with {len(prompts)} prompts\n")

# Encode all text prompts once (for efficiency)
print("Encoding text prompts...")
texts = tokenizer(prompts).to(device)
with torch.no_grad():
    class_embeddings = model.encode_text(texts)
    class_embeddings = F.normalize(class_embeddings, dim=-1)  # [num_prompts, d]
print(f"Text embeddings shape: {class_embeddings.shape}\n")

# Process each image
for img_idx, img_path in enumerate(image_files):
    img_name = os.path.basename(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    print(f"[{img_idx+1}/{len(image_files)}] Processing: {img_name}")
    
    try:
        # Load and preprocess image
        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = preprocess(image_pil)[np.newaxis, :, :, :]
        
        # Store heatmaps for both methods
        heatmaps_original = []
        heatmaps_sparse = []
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            class_name = classes[prompt_idx]
            
            # --- Original TextSpan ---
            original_emb = class_embeddings[prompt_idx:prompt_idx+1]  # [1, d]
            heatmap_orig = compute_heatmap_from_embedding(model, prs, image_tensor, original_emb, device)
            heatmaps_original.append(heatmap_orig)
            
            # --- Sparse TextSpan (with OMP) ---
            # Build dictionary: all classes except current, filter by similarity
            D, kept_indices = build_dictionary_for_class(class_embeddings, prompt_idx, similarity_threshold=0.9)
            
            print(f"  Prompt '{prompt}' (class: {class_name}):")
            print(f"    Dictionary size: {D.shape[0]} atoms")
            if len(kept_indices) > 0:
                kept_classes = [classes[i] for i in kept_indices]
                print(f"    Kept classes: {kept_classes}")
            else:
                print(f"    Kept classes: []")
            
            # Apply OMP to get sparse residual
            sparse_emb = omp_sparse_residual(original_emb, D, max_atoms=8, tol=1e-6)
            heatmap_sparse = compute_heatmap_from_embedding(model, prs, image_tensor, sparse_emb, device)
            heatmaps_sparse.append(heatmap_sparse)
        
        # Compute global min/max for RELATIVE normalization
        # For original heatmaps
        all_original = np.array(heatmaps_original)
        v_min_orig = all_original.min()
        v_max_orig = all_original.max()
        print(f"  Original heatmaps - min: {v_min_orig:.4f}, max: {v_max_orig:.4f}")
        
        # For sparse heatmaps
        all_sparse = np.array(heatmaps_sparse)
        v_min_sparse = all_sparse.min()
        v_max_sparse = all_sparse.max()
        print(f"  Sparse heatmaps - min: {v_min_sparse:.4f}, max: {v_max_sparse:.4f}")
        
        # Create comparison figure
        # Layout: 2 rows (original, sparse) x num_prompts columns
        fig, axes = plt.subplots(2, len(prompts), figsize=(4*len(prompts), 8))
        fig.suptitle(f'Comparison (RELATIVE): {img_name}', fontsize=16, fontweight='bold', y=0.98)
        
        # Row 1: Original TextSpan (using global min/max for this row)
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_original)):
            overlay = create_overlay(image_pil, heatmap, v_min_orig, v_max_orig, alpha=0.5)
            axes[0, col].imshow(overlay)
            axes[0, col].set_title(f'Original\n"{prompt}"', fontsize=10, fontweight='bold')
            axes[0, col].axis('off')
        
        # Row 2: Sparse TextSpan (using global min/max for this row)
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_sparse)):
            overlay = create_overlay(image_pil, heatmap, v_min_sparse, v_max_sparse, alpha=0.5)
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f'Sparse (OMP)\n"{prompt}"', fontsize=10, fontweight='bold')
            axes[1, col].axis('off')
        
        plt.tight_layout()
        
        # Save comparison figure
        output_file = os.path.join(output_path, f'comparison_{img_name_no_ext}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}\n")
        
    except Exception as e:
        print(f"  ✗ Error processing {img_name}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print(f"All done! Comparison heatmaps saved to: {output_path}")
print(f"{'='*80}")
print("\nExplanation:")
print("- Original TextSpan: Uses text embeddings directly")
print("- Sparse TextSpan (OMP): Applies Orthogonal Matching Pursuit to denoise text embeddings")
print("- Dictionary for each class: Contains all other classes (filtered by cosine similarity < 0.9)")
print("- RED regions: HIGH attention for the prompt")
print("- BLUE regions: LOW attention for the prompt")
print("- RELATIVE MODE: All prompts in a row are normalized together using global min/max")
print("  This allows comparing which prompts get higher/lower attention overall")

