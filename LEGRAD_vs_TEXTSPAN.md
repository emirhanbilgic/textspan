# 🔬 LeGrad vs CLIP Text-Based Decomposition (TextSpan)

## 📚 Genel Bakış

Her iki yöntem de **Vision Transformer (ViT)** modellerinde **hangi görüntü bölgelerinin önemli** olduğunu göstermek için attention map'leri kullanıyor. **AMA** tamamen farklı yaklaşımlar!

---

## 🆚 Temel Fark

| Özellik | **LeGrad** | **TextSpan (Bu Makale)** |
|---------|------------|--------------------------|
| **Yöntem** | Gradient-based | Forward-pass based |
| **Backpropagation** | ✅ Gerekli | ❌ Gerekli değil |
| **Text Kullanımı** | Sadece final skor için | Attention'larla direkt align |
| **Hesaplama** | ∇A (gradient) | A @ T (dot product) |
| **Interpreability** | "Hangi attention output'u etkiliyor?" | "Hangi attention text'le uyumlu?" |

---

## 📊 LeGrad Nasıl Çalışıyor?

### **Adım 1: Forward Pass**
```
Image → ViT → Prediction score s^l
```

### **Adım 2: Backward Pass (GRADIENT!)**
```
∇A^l = ∂s/∂A^l  (Attention map'e göre gradient)
```

**Ne yapıyor?**
- Model'in final prediction'ını (örn: "dog" class skoru) maksimize etmek için
- Attention map'in **hangi değerlerinin değişmesi gerektiğini** hesaplıyor
- Yani: "Bu attention değeri artarsa, dog skoru nasıl değişir?"

### **Adım 3: ReLU + Average**
```
E^l(s) = (1/h·n) Σ_h Σ_i (∇A^l_{h,i,.})^+
```

**Ne yapıyor?**
- Negatif gradient'leri at (sadece pozitif etki)
- Tüm head'ler ve patch'ler üzerinden average al

### **Adım 4: Multi-Layer Aggregation**
```
E = norm(reshape(1/L Σ_l E^l))
```

**Sonuç:** Final prediction'ı **en çok etkileyen** attention pattern'leri

---

## 🎯 TextSpan (Bu Makale) Nasıl Çalışıyor?

### **Adım 1: Forward Pass (Attention Çıktıları)**
```
Image → ViT → Attention outputs A^l
Text  → TextEncoder → Text embedding T
```

### **Adım 2: Projected Residual Stream (PRS)**
```
attentions[l, n, h, d]  (her layer, patch, head için)
```

**Ne yapıyor?**
- Her attention head'in **output vektörünü** direkt alıyor
- Gradient yok! Sadece forward pass çıktıları

### **Adım 3: Aggregation (Toplama)**
```
A_total = Σ_layers Σ_heads attentions[l, n, h, :]
         → [256 patches, 1024 dim]
```

**Ne yapıyor?**
- Tüm layer'ların ve head'lerin katkısını toplayarak
- Her patch için **toplam representation vektörü** elde ediyor

### **Adım 4: Text Alignment (DOT PRODUCT)**
```
score[n, c] = A_total[n, :] @ T[c, :]
            = Σ_d (A[n,d] × T[c,d])
```

**Ne yapıyor?**
- Her patch'in vektörü ile text embedding'i arasında **cosine similarity**
- "Bu patch'in representation'ı bu text prompt'uyla ne kadar benzer?"

### **Adım 5: Reshape + Normalize**
```
heatmap = reshape(score) → [16, 16] → [224, 224]
```

**Sonuç:** Her patch'in text prompt'uyla **ne kadar semantically aligned** olduğu

---

## 🔍 Kritik Farklar

### **1. Gradient vs Forward Pass**

#### LeGrad:
```python
# BACKWARD PASS gerekli!
prediction = model(image)
gradient = torch.autograd.grad(prediction, attention_maps)
heatmap = process(gradient)
```

**Soruyor:** "Model prediction'ını değiştirmek için attention'ı nasıl değiştirmeli?"

#### TextSpan:
```python
# Sadece FORWARD PASS!
attentions = model.encode_image(image)  # Attention çıktılarını al
text_emb = model.encode_text(text)      # Text'i encode et
heatmap = attentions @ text_emb.T       # Dot product!
```

**Soruyor:** "Bu patch zaten bu text'le ne kadar uyumlu?"

---

### **2. Text'in Rolü**

#### LeGrad:
- Text sadece **final classification** için kullanılır
- "dog" class'ının gradient'ini hesapla
- Text embeddingi direkt kullanılmıyor

#### TextSpan:
- Text **direkt attention'larla align** ediliyor
- Text embedding attention space'inde yaşıyor
- CLIP'in contrastive learning'inden faydalanıyor

---

### **3. Interpreability**

#### LeGrad: **"Bu attention değeri prediction'ı ETKİLİYOR"**
```
Yüksek gradient → Bu attention output'u değişirse, prediction çok değişir
Düşük gradient  → Bu attention output'u değişirse, prediction az değişir
```

**Analoji:** "Bu tuğlayı çıkarırsam, bina ne kadar sallanır?"

#### TextSpan: **"Bu attention değeri text'le UYUMLU"**
```
Yüksek skor → Bu patch'in representation'ı text embedding'e yakın
Düşük skor  → Bu patch'in representation'ı text embedding'den uzak
```

**Analoji:** "Bu tuğla zaten istediğim renkte mi?"

---

## 📐 Matematiksel Karşılaştırma

### **LeGrad Formülü:**
```
E^l(s) = (1/h·n) Σ_h Σ_i (∂s/∂A^l_{h,i,.})^+
```

**Neler var:**
- `∂s/∂A` → Gradient (backprop gerekli)
- `(.)^+` → ReLU (negatif gradient'leri at)
- Average over heads and patches

### **TextSpan Formülü:**
```
score[n, c] = (Σ_l Σ_h A^l_{n,h,:}) · T_c
```

**Neler var:**
- `A^l_{n,h,:}` → Attention output (forward pass)
- `· T_c` → Dot product with text
- Sum over layers and heads

---

## 🎨 Görselleştirme Farkları

### **LeGrad:**
```
Köpek resmi + "dog" class:
  → Hangi bölgeler "dog" skorunu EN ÇOK artırıyor?
  → Köpeğin ayırt edici özellikleri: baş, kulaklar, burun
```

**Odak:** Discriminative features (ayırt edici)

### **TextSpan:**
```
Köpek resmi + "a photo of a dog" prompt:
  → Hangi bölgeler "dog" text embedding'iyle EN BENZER?
  → Köpeğin tüm bölgeleri: baş, gövde, bacaklar, kuyruk
```

**Odak:** Semantic alignment (anlamsal uyum)

---

## 🧪 Avantajlar ve Dezavantajlar

### **LeGrad**

#### ✅ Avantajlar:
- **Karar mekanizmasını** açıklıyor
- Model hangi feature'lara **karar verirken** bakıyor?
- Class-specific (her class için özelleştirilmiş)

#### ❌ Dezavantajlar:
- Backpropagation gerekli (yavaş)
- Gradient hesabı kararsız olabilir
- Text'i direkt kullanmıyor

### **TextSpan**

#### ✅ Avantajlar:
- **Semantic understanding** gösteriyor
- Text-image alignment direkt ölçülüyor
- Gradient yok → daha hızlı, daha stabil
- CLIP'in zero-shot gücünden faydalanıyor
- **Açık-uçlu text prompt'lar** kullanabilir

#### ❌ Dezavantajlar:
- Final prediction'dan bağımsız (model yanlış tahmin etse bile)
- Discriminative değil, semantic
- CLIP'e özel (başka modellerde direkt çalışmaz)

---

## 🎯 Ne Zaman Hangisi?

### **LeGrad Kullan:**
```
❓ "Model NEDEN bu karar verdi?"
❓ "Model hataysa, hangi bölgeye yanlış baktı?"
❓ "Discriminative feature'lar neler?"
```

**Örnek:** Medical imaging'de yanlış tanı analizi

### **TextSpan Kullan:**
```
❓ "Model bu kavramı görüntüde NEREDE görüyor?"
❓ "Text-image alignment nasıl?"
❓ "Zero-shot olarak yeni kavramları test etmek istiyorum"
```

**Örnek:** "a dog with brown fur" vs "a dog with spots" gibi detaylı prompt'ları karşılaştırma

---

## 💡 TextSpan'in Asıl İnovasyonu

### **1. Attention Decomposition:**
Her layer ve head'in katkısını **ayrı ayrı** görebiliyoruz:
```python
attentions[layer=12, head=3] @ text  # Layer 12, Head 3'ün katkısı
```

### **2. Text-Based Interpretation:**
Gradient yerine **semantic similarity** kullanıyor:
```python
"a photo of a dog" → köpek bölgeleri
"a photo of a golden retriever" → sadece golden retriever özellikleri
"a dog's tail" → kuyruk
```

### **3. Zero-Shot Flexibility:**
Herhangi bir text prompt ile test edebilirsin:
```python
"a photo of a happy dog"
"a photo of a sad dog"
"a dog playing"
"a dog sleeping"
```

---

## 📊 Karşılaştırmalı Örnek

**Köpek Resmi:**

### LeGrad Çıktısı:
```
Prompt: "dog" class
Heatmap: Köpeğin BAŞI çok kırmızı (en discriminative)
         Gövde orta
         Bacaklar düşük
         Arka plan mavi
         
→ Model bu bölgelere bakarak "dog" olduğuna karar veriyor
```

### TextSpan Çıktısı:
```
Prompt: "a photo of a dog"
Heatmap: Köpeğin TÜM BÖLÜMÜ kırmızı/turuncu
         Baş, gövde, bacaklar hepsi yüksek
         Arka plan mavi
         
→ Köpeğin tüm bölgeleri "dog" text embedding'iyle uyumlu
```

---

## 🔬 Kod Karşılaştırması

### **LeGrad:**
```python
# Forward
prediction = model(image)
score = prediction[class_idx]  # "dog" class

# Backward
score.backward()  # GRADIENT!
gradients = attention_map.grad

# Process
heatmap = process_gradients(gradients)  # ReLU, average, etc.
```

### **TextSpan:**
```python
# Forward only
attentions = model.encode_image(image)  # [layers, patches, heads, dim]
text_emb = model.encode_text("a photo of a dog")  # [1024]

# Direct dot product
heatmap = attentions.sum(layers, heads) @ text_emb
# [256 patches, 1024] @ [1024] = [256]
```

---

## ✅ Sonuç

### **LeGrad:**
- **Gradient-based** attribution
- "Model decision'ını etkileyen attention'lar"
- Backprop gerekli

### **TextSpan:**
- **Semantic alignment** based
- "Text ile uyumlu görsel bölgeler"
- Forward pass only

### **Bu Makalenin Asıl Gücü:**
CLIP'in **text-image alignment**'ını kullanarak:
1. Gradient'e gerek kalmadan
2. Herhangi bir text prompt'la
3. Model'in görsel-semantik representation'ını görebiliyoruz

**Ve en önemlisi:** Her attention head ve layer'ın ne yaptığını **text bazlı olarak yorumlayabiliyoruz!**

---

## 📖 Kaynaklar

- **LeGrad:** "LeGrad: Layer-wise Gradient-based Attribution for Vision Transformers"
- **TextSpan:** "Interpreting CLIP's Image Representation via Text-Based Decomposition" (Gandelsman et al., ICLR 2024)

