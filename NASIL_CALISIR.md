# 🧠 CLIP Heatmap Sistemi Nasıl Çalışıyor?

## 📚 Genel Bakış

Bu sistem, CLIP (Contrastive Language-Image Pre-training) modelinin bir görüntüyü anlarken **nereye baktığını** gösteriyor. Her prompt için modelin farklı bölgelere odaklandığını görebiliyoruz.

---

## 🔬 Adım Adım Süreç

### 1️⃣ **Görüntüyü Model İçinden Geçirme**

```python
representation = model.encode_image(image.to(device), 
                                  attn_method='head', 
                                  normalize=False)
attentions, mlps = prs.finalize(representation)
```

**Ne oluyor?**
- Görüntü 224×224 boyutuna resize ediliyor
- ViT (Vision Transformer) 16×16 patch'lere bölüyor = **256 patch**
- Model 24 layer'dan geçiyor (ViT-L-14 için)
- Her layer'da **16 attention head** var
- Her head, her patch'in diğer patch'lerle nasıl "ilişkili" olduğunu öğreniyor

**Attention Shape:**
```
attentions shape: [1, 24, 257, 16, 1024]
                   │   │   │    │   └─ embedding dimension (1024)
                   │   │   │    └───── 16 attention heads
                   │   │   └────────── 257 tokens (1 CLS + 256 patches)
                   │   └────────────── 24 layers
                   └────────────────── batch size (1)
```

### 2️⃣ **Text Prompt'ları Encode Etme**

```python
texts = tokenizer(prompts).to(device)
class_embeddings = model.encode_text(texts)
class_embedding = F.normalize(class_embeddings, dim=-1)
```

**Ne oluyor?**
- Her prompt (örn: "a photo of a dog") CLIP'in text encoder'ından geçiyor
- Her prompt bir **1024-boyutlu vektör** haline geliyor
- 5 prompt = 5 vektör

**Çıkan Shape:**
```
class_embedding shape: [5, 1024]
                        │   └─ embedding dimension
                        └───── 5 different prompts
```

### 3️⃣ **EN ÖNEMLİ ADIM: Attention × Text Benzerliği**

```python
attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
```

Bu satır ÇOK ÖNEMLĐ! Parçalayalım:

#### 🔹 **Adım A:** `attentions[0, :, 1:, :]`
- `[0]` → batch'den 1. görüntü
- `[:, 1:, :]` → CLS token'ı atla (index 0), sadece 256 image patch'ini al
- Shape: `[24, 256, 16, 1024]`

#### 🔹 **Adım B:** `.sum(axis=(0,2))`
- `axis=0` → 24 layer'ı topla (tüm layer'ların katkısı)
- `axis=2` → 16 attention head'i topla (tüm head'lerin katkısı)
- Shape: `[256, 1024]`
- Yani: Her patch için 1024-boyutlu bir "özet vektör"

#### 🔹 **Adım C:** `@ class_embedding.T`
- Matrix çarpımı (dot product)
- `[256, 1024] @ [1024, 5]` = `[256, 5]`
- Her patch için her prompt'la **benzerlik skoru** hesaplanıyor!

**Sonuç:** 256 patch × 5 prompt = Her patch'in her prompt'la ne kadar "uyumlu" olduğu

### 4️⃣ **Spatial Heatmap'e Dönüştürme**

```python
attention_map = F.interpolate(
    einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
    scale_factor=model.visual.patch_size[0],
    mode='bilinear'
).to(device)
```

**Ne oluyor?**
1. `[256, 5]` → `[1, 5, 16, 16]` reshape (16×16 grid'e dönüşüyor)
2. `scale_factor=14` ile 224×224'e upscale (bilinear interpolation)
3. Son shape: `[1, 5, 224, 224]`

### 5️⃣ **Normalizasyon ve Görselleştirme**

```python
v = attention_map[idx] - np.mean(attention_map, axis=0)
```

**ÇOK ÖNEMLĐ:** Bu satır **"relative attention"** hesaplıyor!

- Her prompt'un attention'ı, **ortalamaya göre normalize ediliyor**
- Yani: "Bu prompt için model ORTALAMAYA GÖRE NE KADAR FAZLA/AZ bakıyor?"
- Pozitif değer → Model bu prompt için bu bölgeye FAZLA bakıyor → KIRMIZI
- Negatif değer → Model bu prompt için bu bölgeye AZ bakıyor → MAVĐ

```python
v_normalized = (v - v_min) / (v_max - v_min)
v_uint8 = np.uint8(v_normalized * 255)
heatmap_colored = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
```

- Min-max normalizasyonu → [0, 1] aralığına
- 0-255 aralığına çevir
- JET colormap uygula (MAVİ → KIRMIZI spektrumu)

---

## 🤔 Prompt'lar Birbirini Etkiliyor mu?

### ✅ **EVET ve HAYIR!**

#### 1️⃣ **Model Çalıştırma Aşamasında: HAYIR**
```python
# Her prompt AYRI AYRI encode ediliyor, birbirini ETKİLEMİYOR
for each_prompt in prompts:
    encoding = model.encode_text(each_prompt)  # Bağımsız!
```

Her prompt bağımsız olarak encode ediliyor. Model bir prompt'u işlerken diğerlerinden habersiz.

#### 2️⃣ **Görselleştirme Aşamasında: EVET**
```python
v = attention_map[idx] - np.mean(attention_map, axis=0)
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                         TÜM prompt'ların ortalaması!
```

Heatmap'i oluştururken **tüm prompt'ların ortalamasına göre** normalize ediyoruz!

**Örnek:**
- Eğer görüntüde bir köpek varsa:
  - "a photo of a dog" → Köpek bölgesi YÜKSEK skor (ortalamadan FAZLA)
  - "a photo of a car" → Köpek bölgesi DÜŞÜK skor (ortalamadan AZ)

Bu yüzden:
- 🔴 **KIRMIZI** = Model bu prompt için bu bölgeye ODAKLANMIŞ
- 🔵 **MAVĐ** = Model bu prompt için bu bölgeyi GÖRMEZDEN GELMĐŞ

---

## 🎯 Matematiksel Özet

1. **Image Encoding:** `I → ViT → attentions[24, 256, 16, 1024]`
2. **Text Encoding:** `T → TextEncoder → embeddings[5, 1024]`
3. **Aggregation:** `attentions.sum(layers, heads) → [256, 1024]`
4. **Similarity:** `[256, 1024] @ [1024, 5] → [256, 5]`
5. **Reshape:** `[256, 5] → [16, 16, 5] → [224, 224, 5]`
6. **Absolute Score:** `score[i] → absolute_attention` (her prompt bağımsız)
7. **Visualization:** `absolute_attention → COLORMAP → HEATMAP`

---

## 🔍 Neden Bu Kadar İyi Çalışıyor?

### CLIP'in Gücü:
1. **400 milyon** görüntü-text çifti ile eğitilmiş
2. **Contrastive Learning:** Doğru görüntü-text çiftleri yakın, yanlışlar uzak
3. **Vision Transformer:** Her patch diğer patch'lerle "konuşuyor" (attention)
4. **Multi-Head Attention:** 16 farklı bakış açısı aynı anda

### PRS (Projected Residual Stream) Metodu:
Bu projede kullanılan özel teknik:
- Her layer'ın katkısını ayrı ayrı izliyor
- Her attention head'in ne yaptığını görebiliyoruz
- Bu sayede "model nereye bakıyor?" sorusunu cevaplayabiliyoruz

---

## 💡 Sonuç

**Heatmap'ler şunu gösteriyor:**
> Model bir görüntüyü görünce, verilen prompt'a göre görüntünün **hangi bölgelerinin o prompt'la uyumlu olduğunu** hesaplıyor.

- "a photo of a dog" → Köpek olan bölgeler aktivasyonu artıyor → KIRMIZI
- "a photo of a car" → Araba olan bölgeler aktivasyonu artıyor → KIRMIZI

Her prompt için model **bağımsız bir karar veriyor**, ama görselleştirmede hepsini **karşılaştırmalı olarak** gösteriyoruz!

---

## 📖 Kaynaklar

Bu proje şu makaleye dayanıyor:
- **Paper:** "Interpreting CLIP's Image Representation via Text-Based Decomposition"
- **Authors:** Yossi Gandelsman, Alexei A. Efros, Jacob Steinhardt
- **Conference:** ICLR 2024

