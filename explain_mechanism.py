"""
🎓 CLIP Heatmap Mekanizması - Görsel Açıklama
Bu script, kodun nasıl çalıştığını ADIM ADIM gösterir
"""

import numpy as np

print("="*80)
print("🧠 CLIP HEATMAP MEKANĐZMASINI ANLAMAK")
print("="*80)

print("\n" + "─"*80)
print("📸 ADIM 1: GÖRÜNTÜ İŞLEME")
print("─"*80)

print("""
Bir köpek resmi yüklüyoruz:
  🐕 dog.jpeg (original size: 1920×1080)
  
Preprocessing:
  → Resize: 224×224
  → Normalize: mean=[0.48145466, 0.4578275, 0.40821073]
  → Tensor: shape [1, 3, 224, 224]
""")

print("\n" + "─"*80)
print("🔲 ADIM 2: VISION TRANSFORMER (ViT) - PATCH'LERE BÖLME")
print("─"*80)

image_size = 224
patch_size = 14
num_patches = (image_size // patch_size) ** 2

print(f"""
ViT görüntüyü {patch_size}×{patch_size} patch'lere böler:
  
  224×224 görüntü → {image_size // patch_size}×{image_size // patch_size} grid = {num_patches} patches
  
Görsel olarak:
  ┌─────────────────────────────────┐
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │  ← 16 patches
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ 🐕🐕🐕🐕 □ □ □ □ □ □ □ │  ← Köpek burada!
  │ □ □ □ 🐕🐕🐕🐕🐕🐕 □ □ □ □ □ │
  │ □ □ □ 🐕🐕🐕🐕🐕🐕 □ □ □ □ □ │
  │ □ □ □ □ 🐕🐕🐕🐕 □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  │ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ │
  └─────────────────────────────────┘
  
Her patch → 1024 boyutlu embedding vector
+ 1 CLS token (class token) = 257 total tokens
""")

print("\n" + "─"*80)
print("🔗 ADIM 3: MULTI-HEAD ATTENTION")
print("─"*80)

num_layers = 24
num_heads = 16
embed_dim = 1024

print(f"""
ViT-L-14 modeli:
  • {num_layers} Transformer layer
  • Her layer'da {num_heads} attention head
  • Embedding dimension: {embed_dim}
  
Attention nedir?
  → Her patch diğer patch'lerle "konuşur"
  → "Sen benimle ne kadar alakalısın?" diye sorar
  → Köpek patch'leri birbirini "tanır", arka plan'ı ignore eder
  
Örnek attention pattern (Layer 12, Head 3):
  
  Patch[64] (köpeğin başı):
    ├─ Patch[64] (kendisi)     → 0.25 (yüksek attention!)
    ├─ Patch[65] (köpek gövde)  → 0.18 (yüksek!)
    ├─ Patch[80] (köpek bacak)  → 0.12 (orta)
    ├─ Patch[10] (arka plan)    → 0.01 (düşük!)
    └─ Patch[15] (arka plan)    → 0.01 (düşük!)
""")

# Simulate attention scores
print("\n" + "─"*80)
print("📊 ADIM 4: ATTENTION TOPLAMA")
print("─"*80)

print("""
Kod:
  attentions[0, :, 1:, :].sum(axis=(0,2))
  
Ne yapıyor?
  • [0] → Batch'den ilk görüntü
  • [:, 1:, :] → CLS token'ı atla, sadece 256 image patch
  • .sum(axis=0) → 24 layer'ı TOPLA
  • .sum(axis=2) → 16 head'i TOPLA
  
Sonuç:
  [256 patches, 1024 dim] → Her patch'in "toplam attention vektörü"
  
Örnek:
  Patch[64] → [0.12, -0.05, 0.31, ..., 0.08]  (1024 sayı)
  Patch[65] → [0.15, -0.02, 0.28, ..., 0.11]
  ...
""")

print("\n" + "─"*80)
print("💬 ADIM 5: TEXT ENCODING")
print("─"*80)

prompts = [
    'a photo of a car',
    'a photo of a plane', 
    'a photo of a bird',
    'a photo of a cat',
    'a photo of a dog'
]

print(f"""
{len(prompts)} prompt'u encode ediyoruz:
""")

for i, prompt in enumerate(prompts, 1):
    print(f'  {i}. "{prompt}"')
    print(f'     → Text Encoder → [{embed_dim}d vector]')
    if i < len(prompts):
        print()

print("""
Text encoder çıktısı:
  shape: [5 prompts, 1024 dim]
  
  Normalize ediliyor (L2 norm = 1):
  embedding = embedding / ||embedding||
""")

print("\n" + "─"*80)
print("🎯 ADIM 6: BENZERLİK HESAPLAMA (EN ÖNEMLİ!)")
print("─"*80)

print("""
Kod:
  attention_map = attentions @ class_embedding.T
  
Matrix çarpımı:
  [256 patches, 1024 dim] @ [1024 dim, 5 prompts]
  = [256 patches, 5 prompts]
  
Bu NE DEMEK?
  → Her patch için, her prompt'la benzerlik skoru!
  
Görsel örnek (köpek resmi için):
""")

# Simulated similarity scores
dog_patches = [64, 65, 80, 81, 96, 97]
car_prompt_idx = 0
dog_prompt_idx = 4

print("""
                    car    plane   bird    cat     dog
                    ────   ─────   ────    ───     ───
  Patch[10] (bg)    0.12   0.08    0.11    0.09    0.10  ← Arka plan, hepsi düşük
  Patch[64] (🐕)    0.15   0.11    0.18    0.45    0.78  ← Köpek! "dog" en yüksek!
  Patch[65] (🐕)    0.13   0.09    0.20    0.42    0.75  ← Köpek! "dog" yüksek!
  Patch[80] (🐕)    0.14   0.10    0.19    0.48    0.81  ← Köpek! "dog" en yüksek!
  Patch[150](bg)    0.11   0.07    0.10    0.08    0.09  ← Arka plan
  
🔍 Dikkat: Köpek patch'leri "dog" prompt'u için yüksek skor alıyor!
""")

print("\n" + "─"*80)
print("📐 ADIM 7: SPATIAL RESHAPE")
print("─"*80)

print("""
[256, 5] tensor'ı → [16, 16, 5] grid'e reshape ediyoruz
Sonra bilinear interpolation ile 224×224'e büyütüyoruz

  [256, 5]  →  reshape  →  [1, 5, 16, 16]  →  interpolate  →  [1, 5, 224, 224]
  
Artık her piksel için, her prompt'un skoru var!
""")

print("\n" + "─"*80)
print("🎨 ADIM 8: NORMALIZASYON VE GÖRSELLEŞTĐRME")
print("─"*80)

# Simulate scores
np.random.seed(42)
car_scores = np.random.uniform(0.1, 0.3, 256)
car_scores[50:70] = np.random.uniform(0.6, 0.9, 20)  # Araba bölgesi (varsayımsal)

dog_scores = np.random.uniform(0.1, 0.3, 256)
dog_scores[60:85] = np.random.uniform(0.7, 0.95, 25)  # Köpek bölgesi

all_scores = np.stack([car_scores, dog_scores])
mean_scores = all_scores.mean(axis=0)

print("""
ÖNEMLİ: Relative Normalization!

  relative_score[i] = score[i] - mean(all_scores)
  
Neden?
  → Mutlak skor değil, ORTALAMAYA GÖRE fark önemli!
  → "Bu prompt için model diğerlerine GÖRE ne kadar fazla bakıyor?"
  
Örnek (Patch[64] - köpeğin başı):
""")

patch_idx = 64
print(f"""
  "a photo of a car"  → score: 0.15 → relative: 0.15 - 0.35 = -0.20 (MAVİ!)
  "a photo of a dog"  → score: 0.78 → relative: 0.78 - 0.35 = +0.43 (KIRMIZI!)
                                                              ^^^^^^^^
                                                              Ortalama: 0.35
""")

print("""
Min-Max normalization:
  normalized = (score - min) / (max - min)  → [0, 1] aralığı
  uint8 = normalized * 255                  → [0, 255] aralığı
  
Colormap (JET):
  0   → 🔵 MAVİ   (düşük attention)
  127 → 🟢 YEŞİL  (orta)
  255 → 🔴 KIRMIZI (yüksek attention)
""")

print("\n" + "─"*80)
print("🤔 PROMPT'LAR BİRBİRİNİ ETKİLİYOR MU?")
print("─"*80)

print("""
CEVAP: Kısmen EVET!

1️⃣ Model Inference Sırasında: HAYIR ❌
   • Her prompt AYRI encode ediliyor
   • Model bir prompt'u işlerken diğerlerinden habersiz
   • Bağımsız skorlar hesaplanıyor

2️⃣ Görselleştirme Sırasında: EVET ✅
   • Relative normalization kullanıyoruz
   • Her prompt'un skoru, TÜM prompt'ların ortalamasına göre
   • Bu yüzden:
     - Bir prompt ekler/çıkarırsanız → Renkler DEĞİŞİR
     - Ama model'in GERÇEK skoru aynı kalır
     
Örnek:
  Eğer sadece "a photo of a dog" prompt'unu kullansaydık:
    → Köpek bölgesi yine yüksek skor alırdı
    → Ama GÖRSELDE her yer orta-yüksek renkte olurdu
    → Çünkü karşılaştıracak başka prompt yok!
    
  5 prompt kullanınca:
    → Her prompt diğerleriyle KIYASLANIR
    → "dog" prompt'u köpek bölgesinde DAHA FAZLA aktivasyon → KIRMIZI
    → "car" prompt'u köpek bölgesinde DAHA AZ aktivasyon → MAVİ
""")

print("\n" + "─"*80)
print("💡 ÖZET")
print("─"*80)

print("""
Heatmap'ler şunu gösterir:
  
  🔴 KIRMIZI = Model bu prompt için bu bölgeye ODAKLANMIŞ
  🔵 MAVİ   = Model bu prompt için bu bölgeyi YOK SAYMIŞ
  
Nasıl?
  1. Görüntü → 256 patch'e bölünüyor
  2. Her patch → 24 layer × 16 head attention geçiyor
  3. Text prompt → 1024d vektör oluyor
  4. Her patch × her prompt → benzerlik skoru
  5. Ortalamaya göre normalize → relative attention
  6. Colormap → görselleştirme
  
CLIP'in gücü:
  • 400M görüntü-text çifti ile eğitilmiş
  • "Köpek" kelimesi → köpek görüntü feature'ları ile align
  • Bu yüzden "a photo of a dog" prompt'u köpek patch'lerini aktive ediyor!
""")

print("\n" + "="*80)
print("✅ AÇIKLAMA TAMAMLANDI!")
print("="*80)
print("\nDaha fazla detay için: NASIL_CALISIR.md dosyasını oku!\n")

