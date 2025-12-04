#!/usr/bin/env python
"""
Comprehensive comparison of TextSpan vs Sparse TextSpan with 7 dictionary configurations.
Each configuration tested with both RELATIVE and INDEPENDENT normalization modes.
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
from nltk.corpus import wordnet as wn

# Configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pretrained = 'laion2b_s34b_b88k'
model_name = 'ViT-B-16'

# Use paths relative to this file so it runs both locally and on Kaggle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data')
base_output_path = os.path.join(BASE_DIR, 'comparison_output')

print(f"Using device: {device}")

# ============================================================================
# WordNet Utilities
# ============================================================================

def get_wordnet_synsets(word):
    """Get all synsets for a word."""
    return wn.synsets(word.replace('_', ' ').replace('-', ' '))

def get_hypernyms(word):
    """Get hypernyms for a word from WordNet."""
    synsets = get_wordnet_synsets(word)
    hypernyms = set()
    for synset in synsets:
        for hypernym in synset.hypernyms():
            # Get lemma names
            for lemma in hypernym.lemmas():
                hypernyms.add(lemma.name().replace('_', ' '))
    return list(hypernyms)

def get_co_hyponyms(word):
    """Get co-hyponyms (siblings) for a word from WordNet."""
    synsets = get_wordnet_synsets(word)
    co_hyponyms = set()
    
    for synset in synsets:
        # Get hypernyms first
        for hypernym in synset.hypernyms():
            # Get all hyponyms of the hypernym (these are siblings)
            for sibling in hypernym.hyponyms():
                if sibling != synset:  # Exclude the word itself
                    for lemma in sibling.lemmas():
                        co_hyponyms.add(lemma.name().replace('_', ' '))
    
    return list(co_hyponyms)

def get_wordnet_related_words(word, max_words=20):
    """Get related words (hypernyms and co-hyponyms) for a word."""
    hypernyms = get_hypernyms(word)
    co_hyponyms = get_co_hyponyms(word)
    
    # Limit to reasonable number
    hypernyms = hypernyms[:max_words]
    co_hyponyms = co_hyponyms[:max_words]
    
    return hypernyms, co_hyponyms

# ============================================================================
# OMP and Dictionary Building
# ============================================================================

def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], atom rows, L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)
        x_hat = (s.t() @ D_S).to(x.dtype)
        r = (x - x_hat)
        if float(torch.norm(r) <= tol):
            break
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)

def encode_text_list(model, tokenizer, text_list, device):
    """Encode a list of text strings to embeddings."""
    if not text_list:
        return None
    texts = tokenizer(text_list).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(texts)
        embeddings = F.normalize(embeddings, dim=-1)
    return embeddings

def build_dictionary_config_1_prompts_only(class_embeddings, target_idx, classes, model, tokenizer, device, 
                                           similarity_threshold=0.9):
    """Config 1: Only other prompts in dictionary."""
    other_indices = [i for i in range(len(class_embeddings)) if i != target_idx]
    if len(other_indices) == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    target_emb = class_embeddings[target_idx:target_idx+1]
    other_embs = class_embeddings[other_indices]
    
    # Filter by similarity
    sims = (other_embs @ target_emb.t()).squeeze(-1).abs()
    keep_mask = sims < similarity_threshold
    
    if keep_mask.sum() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D = other_embs[keep_mask]
    D = F.normalize(D, dim=-1)
    kept_classes = [classes[other_indices[i]] for i, keep in enumerate(keep_mask.tolist()) if keep]
    
    return D, kept_classes

def build_dictionary_config_2_hypernyms_only(class_embeddings, target_idx, classes, model, tokenizer, device,
                                              similarity_threshold=0.9):
    """Config 2: Only hypernyms from WordNet."""
    target_class = classes[target_idx]
    hypernyms, _ = get_wordnet_related_words(target_class)
    
    if not hypernyms:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    # Create prompts from hypernyms
    hypernym_prompts = [f'a photo of a {h}' for h in hypernyms]
    hypernym_embs = encode_text_list(model, tokenizer, hypernym_prompts, device)
    
    if hypernym_embs is None:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    # Filter by similarity with target
    target_emb = class_embeddings[target_idx:target_idx+1]
    sims = (hypernym_embs @ target_emb.t()).squeeze(-1).abs()
    keep_mask = sims < similarity_threshold
    
    if keep_mask.sum() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D = hypernym_embs[keep_mask]
    D = F.normalize(D, dim=-1)
    kept_names = [hypernyms[i] for i, keep in enumerate(keep_mask.tolist()) if keep]
    
    return D, kept_names

def build_dictionary_config_3_cohyponyms_only(class_embeddings, target_idx, classes, model, tokenizer, device,
                                               similarity_threshold=0.9):
    """Config 3: Only co-hyponyms from WordNet."""
    target_class = classes[target_idx]
    _, co_hyponyms = get_wordnet_related_words(target_class)
    
    if not co_hyponyms:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    # Create prompts from co-hyponyms
    cohyponym_prompts = [f'a photo of a {ch}' for ch in co_hyponyms]
    cohyponym_embs = encode_text_list(model, tokenizer, cohyponym_prompts, device)
    
    if cohyponym_embs is None:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    # Filter by similarity with target
    target_emb = class_embeddings[target_idx:target_idx+1]
    sims = (cohyponym_embs @ target_emb.t()).squeeze(-1).abs()
    keep_mask = sims < similarity_threshold
    
    if keep_mask.sum() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D = cohyponym_embs[keep_mask]
    D = F.normalize(D, dim=-1)
    kept_names = [co_hyponyms[i] for i, keep in enumerate(keep_mask.tolist()) if keep]
    
    return D, kept_names

def build_dictionary_config_4_hypernyms_prompts(class_embeddings, target_idx, classes, model, tokenizer, device,
                                                 similarity_threshold=0.9):
    """Config 4: Hypernyms + other prompts."""
    # Get prompts
    D_prompts, kept_prompts = build_dictionary_config_1_prompts_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Get hypernyms
    D_hyper, kept_hyper = build_dictionary_config_2_hypernyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Combine
    if D_prompts.numel() == 0 and D_hyper.numel() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D_list = []
    names_list = []
    if D_prompts.numel() > 0:
        D_list.append(D_prompts)
        names_list.extend([f"prompt:{p}" for p in kept_prompts])
    if D_hyper.numel() > 0:
        D_list.append(D_hyper)
        names_list.extend([f"hyper:{h}" for h in kept_hyper])
    
    D = torch.cat(D_list, dim=0)
    D = F.normalize(D, dim=-1)
    
    return D, names_list

def build_dictionary_config_5_prompts_cohyponyms(class_embeddings, target_idx, classes, model, tokenizer, device,
                                                  similarity_threshold=0.9):
    """Config 5: Prompts + co-hyponyms."""
    # Get prompts
    D_prompts, kept_prompts = build_dictionary_config_1_prompts_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Get co-hyponyms
    D_cohypo, kept_cohypo = build_dictionary_config_3_cohyponyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Combine
    if D_prompts.numel() == 0 and D_cohypo.numel() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D_list = []
    names_list = []
    if D_prompts.numel() > 0:
        D_list.append(D_prompts)
        names_list.extend([f"prompt:{p}" for p in kept_prompts])
    if D_cohypo.numel() > 0:
        D_list.append(D_cohypo)
        names_list.extend([f"cohypo:{c}" for c in kept_cohypo])
    
    D = torch.cat(D_list, dim=0)
    D = F.normalize(D, dim=-1)
    
    return D, names_list

def build_dictionary_config_6_hypernyms_cohyponyms(class_embeddings, target_idx, classes, model, tokenizer, device,
                                                    similarity_threshold=0.9):
    """Config 6: Hypernyms + co-hyponyms."""
    # Get hypernyms
    D_hyper, kept_hyper = build_dictionary_config_2_hypernyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Get co-hyponyms
    D_cohypo, kept_cohypo = build_dictionary_config_3_cohyponyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Combine
    if D_hyper.numel() == 0 and D_cohypo.numel() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D_list = []
    names_list = []
    if D_hyper.numel() > 0:
        D_list.append(D_hyper)
        names_list.extend([f"hyper:{h}" for h in kept_hyper])
    if D_cohypo.numel() > 0:
        D_list.append(D_cohypo)
        names_list.extend([f"cohypo:{c}" for c in kept_cohypo])
    
    D = torch.cat(D_list, dim=0)
    D = F.normalize(D, dim=-1)
    
    return D, names_list

def build_dictionary_config_7_all(class_embeddings, target_idx, classes, model, tokenizer, device,
                                   similarity_threshold=0.9):
    """Config 7: All (hypernyms + co-hyponyms + prompts)."""
    # Get prompts
    D_prompts, kept_prompts = build_dictionary_config_1_prompts_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Get hypernyms
    D_hyper, kept_hyper = build_dictionary_config_2_hypernyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Get co-hyponyms
    D_cohypo, kept_cohypo = build_dictionary_config_3_cohyponyms_only(
        class_embeddings, target_idx, classes, model, tokenizer, device, similarity_threshold)
    
    # Combine all
    if D_prompts.numel() == 0 and D_hyper.numel() == 0 and D_cohypo.numel() == 0:
        return class_embeddings.new_zeros((0, class_embeddings.shape[-1])), []
    
    D_list = []
    names_list = []
    if D_prompts.numel() > 0:
        D_list.append(D_prompts)
        names_list.extend([f"prompt:{p}" for p in kept_prompts])
    if D_hyper.numel() > 0:
        D_list.append(D_hyper)
        names_list.extend([f"hyper:{h}" for h in kept_hyper])
    if D_cohypo.numel() > 0:
        D_list.append(D_cohypo)
        names_list.extend([f"cohypo:{c}" for c in kept_cohypo])
    
    D = torch.cat(D_list, dim=0)
    D = F.normalize(D, dim=-1)
    
    return D, names_list

# Dictionary of all configurations
DICTIONARY_CONFIGS = {
    'config_1_prompts_only': build_dictionary_config_1_prompts_only,
    'config_2_hypernyms_only': build_dictionary_config_2_hypernyms_only,
    'config_3_cohyponyms_only': build_dictionary_config_3_cohyponyms_only,
    'config_4_hypernyms_prompts': build_dictionary_config_4_hypernyms_prompts,
    'config_5_prompts_cohyponyms': build_dictionary_config_5_prompts_cohyponyms,
    'config_6_hypernyms_cohyponyms': build_dictionary_config_6_hypernyms_cohyponyms,
    'config_7_all': build_dictionary_config_7_all,
}

# ============================================================================
# Heatmap Generation
# ============================================================================

def compute_heatmap_from_embedding(model, prs, image_tensor, text_embedding, device):
    """Compute heatmap for a given text embedding."""
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image_tensor.to(device), 
                                          attn_method='head', 
                                          normalize=False)
        attentions, mlps = prs.finalize(representation)
    
    attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ text_embedding.T
    
    attention_map = F.interpolate(
        einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
        scale_factor=model.visual.patch_size[0],
        mode='bilinear'
    ).to(device)
    
    attention_map = attention_map[0, 0].detach().cpu().numpy()
    
    return attention_map

def create_overlay(image_pil, heatmap, v_min, v_max, alpha=0.5):
    """Create overlay of heatmap on image."""
    if v_max - v_min > 0:
        v_normalized = np.clip((heatmap - v_min) / (v_max - v_min), 0, 1)
    else:
        v_normalized = heatmap * 0
    v_uint8 = np.uint8(v_normalized * 255)
    
    heatmap_colored = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_array = np.array(image_pil)
    heatmap_resized = cv2.resize(heatmap_rgb, (img_array.shape[1], img_array.shape[0]))
    
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_resized, alpha, 0)
    
    return overlay

# ============================================================================
# Main Processing
# ============================================================================

def process_image_for_config(img_path, prompts, classes, class_embeddings, model, prs, tokenizer, 
                             config_name, config_func, normalization_mode, output_path, device):
    """Process a single image for a given configuration and normalization mode."""
    img_name = os.path.basename(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    # Load and preprocess image
    image_pil = Image.open(img_path).convert('RGB')
    image_tensor = preprocess(image_pil)[np.newaxis, :, :, :]
    
    # Store heatmaps for both methods
    heatmaps_original = []
    heatmaps_sparse = []
    dictionary_info = []
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts):
        class_name = classes[prompt_idx]
        
        # --- Original TextSpan ---
        original_emb = class_embeddings[prompt_idx:prompt_idx+1]
        heatmap_orig = compute_heatmap_from_embedding(model, prs, image_tensor, original_emb, device)
        heatmaps_original.append(heatmap_orig)
        
        # --- Sparse TextSpan (with OMP) ---
        D, kept_items = config_func(class_embeddings, prompt_idx, classes, model, tokenizer, device)
        
        dictionary_info.append({
            'class': class_name,
            'dict_size': D.shape[0] if D.numel() > 0 else 0,
            'kept_items': kept_items
        })
        
        # Apply OMP to get sparse residual
        sparse_emb = omp_sparse_residual(original_emb, D, max_atoms=8, tol=1e-6)
        heatmap_sparse = compute_heatmap_from_embedding(model, prs, image_tensor, sparse_emb, device)
        heatmaps_sparse.append(heatmap_sparse)
    
    # Normalization based on mode
    if normalization_mode == 'relative':
        # RELATIVE: normalize all prompts together (separately for original and sparse)
        all_original = np.array(heatmaps_original)
        v_min_orig = all_original.min()
        v_max_orig = all_original.max()
        
        all_sparse = np.array(heatmaps_sparse)
        v_min_sparse = all_sparse.min()
        v_max_sparse = all_sparse.max()
        
        # Create comparison figure
        fig, axes = plt.subplots(2, len(prompts), figsize=(4*len(prompts), 8))
        fig.suptitle(f'{config_name} - RELATIVE: {img_name}', fontsize=14, fontweight='bold')
        
        # Row 1: Original TextSpan
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_original)):
            overlay = create_overlay(image_pil, heatmap, v_min_orig, v_max_orig, alpha=0.5)
            axes[0, col].imshow(overlay)
            axes[0, col].set_title(f'Original\n"{prompt}"', fontsize=9)
            axes[0, col].axis('off')
        
        # Row 2: Sparse TextSpan
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_sparse)):
            overlay = create_overlay(image_pil, heatmap, v_min_sparse, v_max_sparse, alpha=0.5)
            axes[1, col].imshow(overlay)
            dict_info = dictionary_info[col]
            axes[1, col].set_title(f'Sparse (OMP)\n"{prompt}"\nDict: {dict_info["dict_size"]} atoms', 
                                  fontsize=9)
            axes[1, col].axis('off')
        
    else:  # independent
        # INDEPENDENT: normalize each heatmap independently
        fig, axes = plt.subplots(2, len(prompts), figsize=(4*len(prompts), 8))
        fig.suptitle(f'{config_name} - INDEPENDENT: {img_name}', fontsize=14, fontweight='bold')
        
        # Row 1: Original TextSpan
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_original)):
            v_min = heatmap.min()
            v_max = heatmap.max()
            overlay = create_overlay(image_pil, heatmap, v_min, v_max, alpha=0.5)
            axes[0, col].imshow(overlay)
            axes[0, col].set_title(f'Original\n"{prompt}"', fontsize=9)
            axes[0, col].axis('off')
        
        # Row 2: Sparse TextSpan
        for col, (prompt, heatmap) in enumerate(zip(prompts, heatmaps_sparse)):
            v_min = heatmap.min()
            v_max = heatmap.max()
            overlay = create_overlay(image_pil, heatmap, v_min, v_max, alpha=0.5)
            axes[1, col].imshow(overlay)
            dict_info = dictionary_info[col]
            axes[1, col].set_title(f'Sparse (OMP)\n"{prompt}"\nDict: {dict_info["dict_size"]} atoms', 
                                  fontsize=9)
            axes[1, col].axis('off')
    
    plt.tight_layout()
    
    # Save comparison figure
    output_file = os.path.join(output_path, f'{img_name_no_ext}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file, dictionary_info

# ============================================================================
# Run All Configurations
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE TEXTSPAN VS SPARSE TEXTSPAN COMPARISON")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer(model_name)
    print(f"Model loaded successfully!")
    
    # Hook the PRS logger
    prs = hook_prs_logger(model, device)
    
    # Define prompts and classes
    prompts = [
        'a photo of a automobile',
        'a photo of a plane', 
        'a photo of a bird',
        'a photo of a cat',
        'a photo of a dog'
    ]
    classes = ['automobile', 'plane', 'bird', 'cat', 'dog']
    
    print(f"\nPrompts: {prompts}")
    print(f"Classes: {classes}")
    
    # Test WordNet functionality
    print("\nTesting WordNet lookups for each class:")
    for cls in classes:
        hypernyms, co_hyponyms = get_wordnet_related_words(cls)
        print(f"  {cls}:")
        print(f"    Hypernyms: {hypernyms[:5]}")
        print(f"    Co-hyponyms: {co_hyponyms[:5]}")
    
    # Encode text prompts
    print("\nEncoding text prompts...")
    texts = tokenizer(prompts).to(device)
    with torch.no_grad():
        class_embeddings = model.encode_text(texts)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
    print(f"Text embeddings shape: {class_embeddings.shape}")
    
    # Get all image files
    image_files = glob.glob(os.path.join(data_path, '*.png')) + \
                  glob.glob(os.path.join(data_path, '*.jpg')) + \
                  glob.glob(os.path.join(data_path, '*.jpeg'))
    print(f"\nFound {len(image_files)} images")
    
    # Process all configurations
    total_images = 0
    for config_name, config_func in DICTIONARY_CONFIGS.items():
        for norm_mode in ['relative', 'independent']:
            output_folder = os.path.join(base_output_path, f'{config_name}_{norm_mode}')
            os.makedirs(output_folder, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"Processing: {config_name} - {norm_mode.upper()}")
            print(f"Output: {output_folder}")
            print(f"{'='*80}")
            
            for img_idx, img_path in enumerate(image_files):
                img_name = os.path.basename(img_path)
                print(f"  [{img_idx+1}/{len(image_files)}] {img_name}...", end=' ')
                
                try:
                    output_file, dict_info = process_image_for_config(
                        img_path, prompts, classes, class_embeddings, 
                        model, prs, tokenizer, config_name, config_func, 
                        norm_mode, output_folder, device
                    )
                    print(f"✓ Saved")
                    total_images += 1
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"ALL DONE!")
    print(f"{'='*80}")
    print(f"Total images generated: {total_images}")
    print(f"Expected: {14 * len(image_files)} = {14 * len(image_files)}")
    print(f"Output directory: {base_output_path}")
    print(f"\nGenerated 14 folders:")
    print(f"  - 7 dictionary configurations")
    print(f"  - 2 normalization modes (relative, independent)")
    print(f"  - Each folder contains {len(image_files)} comparison images")

