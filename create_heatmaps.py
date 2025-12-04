## Generate heatmaps for images in data folder
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

# Configuration
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pretrained = 'laion2b_s32b_b82k'
model_name = 'ViT-L-14'
data_path = '/Users/emirhan/Desktop/clip_text_span/data'
output_path = '/Users/emirhan/Desktop/clip_text_span/heatmaps_output'

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
    'a photo of a car',
    'a photo of a plane', 
    'a photo of a bird',
    'a photo of a cat',
    'a photo of a dog'
]

# Get all image files
image_files = glob.glob(os.path.join(data_path, '*.png')) + \
              glob.glob(os.path.join(data_path, '*.jpg')) + \
              glob.glob(os.path.join(data_path, '*.jpeg'))

print(f"\nFound {len(image_files)} images")
print(f"Testing with {len(prompts)} prompts: {prompts}\n")

# Process each image
for img_path in image_files:
    img_name = os.path.basename(img_path)
    img_name_no_ext = os.path.splitext(img_name)[0]
    
    print(f"Processing: {img_name}")
    
    # Load and preprocess image
    try:
        image_pil = Image.open(img_path).convert('RGB')
        image = preprocess(image_pil)[np.newaxis, :, :, :]
        
        # Run the image through the model
        prs.reinit()
        with torch.no_grad():
            representation = model.encode_image(image.to(device), 
                                              attn_method='head', 
                                              normalize=False)
            attentions, mlps = prs.finalize(representation)
        
        # Encode text prompts
        texts = tokenizer(prompts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1)
        
        # Compute attention maps
        attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
        
        # Interpolate to image size
        attention_map = F.interpolate(
            einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
            scale_factor=model.visual.patch_size[0],
            mode='bilinear'
        ).to(device)
        attention_map = attention_map[0].detach().cpu().numpy()
        
        # Create figure with subplots for each prompt
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Heatmaps for {img_name}', fontsize=16, fontweight='bold')
        
        # Show original image
        axes[0, 0].imshow(image_pil)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Create heatmap for each prompt
        for idx, prompt in enumerate(prompts):
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            
            # Get attention for this prompt (absolute, independent)
            v = attention_map[idx]
            
            # Normalize to 0-255
            v_min, v_max = v.min(), v.max()
            if v_max - v_min > 0:
                v_normalized = (v - v_min) / (v_max - v_min)
            else:
                v_normalized = v * 0
            v_uint8 = np.uint8(v_normalized * 255)
            
            # Apply colormap (JET - blue is low, red is high)
            heatmap_colored = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match image size
            img_array = np.array(image_pil)
            heatmap_resized = cv2.resize(heatmap_rgb, (img_array.shape[1], img_array.shape[0]))
            
            # Overlay heatmap on image
            alpha = 0.5
            overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_resized, alpha, 0)
            
            # Display
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'"{prompt}"', fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_path, f'{img_name_no_ext}_heatmaps.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
        
        # Also save individual heatmaps for each prompt
        for idx, prompt in enumerate(prompts):
            fig_individual, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            ax1.imshow(image_pil)
            ax1.set_title('Original Image', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Pure heatmap (absolute, independent)
            v = attention_map[idx]
            v_min, v_max = v.min(), v.max()
            if v_max - v_min > 0:
                v_normalized = (v - v_min) / (v_max - v_min)
            else:
                v_normalized = v * 0
            v_uint8 = np.uint8(v_normalized * 255)
            heatmap_colored = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            im = ax2.imshow(heatmap_rgb)
            ax2.set_title(f'Heatmap: "{prompt}"', fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # Overlay
            img_array = np.array(image_pil)
            heatmap_resized = cv2.resize(heatmap_rgb, (img_array.shape[1], img_array.shape[0]))
            alpha = 0.5
            overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_resized, alpha, 0)
            
            ax3.imshow(overlay)
            ax3.set_title('Overlay (50%)', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            plt.suptitle(f'{img_name} - "{prompt}"', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save individual
            prompt_safe = prompt.replace(' ', '_')
            output_file_individual = os.path.join(output_path, 
                                                  f'{img_name_no_ext}_{prompt_safe}.png')
            plt.savefig(output_file_individual, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Saved individual heatmaps for all prompts\n")
        
    except Exception as e:
        print(f"  ✗ Error processing {img_name}: {str(e)}\n")
        continue

print(f"\n{'='*60}")
print(f"All done! Heatmaps saved to: {output_path}")
print(f"{'='*60}")
print("\nHeatmap Açıklaması (BAĞIMSIZ MODDA):")
print("- KIRMIZI bölgeler: Model bu prompt için bu bölgelere YÜKSEK attention veriyor")
print("- MAVİ bölgeler: Model bu prompt için bu bölgelere DÜŞÜK attention veriyor")
print("- Her prompt BAĞIMSIZ olarak gösteriliyor (karşılaştırmalı değil)")
print("- Artık tüm prompt'lar için benzer bölgeler benzer renkler gösterebilir")

