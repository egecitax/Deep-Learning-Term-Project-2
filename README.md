# CNN Tabanlı Görüntü Sınıflandırma Modelleri

## Giriş
Bu çalışmada, yapay sinir ağlarının (CNN) görüntü sınıflandırma üzerindeki performansı değerlendirilmiştir. Amaç, farklı CNN mimarilerinin farklı düzenlemelerle (Batch Normalization, Dropout gibi) başarımını kıyaslamak ve bir klasik makine öğrenmesi modeliyle hibrit bir yaklaşım sergilemektir. Çalışma boyunca MNIST ve CIFAR-10 veri setleri kullanılmıştır.

---

## Yöntem

### Kullanılan Veri Setleri
- **MNIST**: 28x28 boyutunda gri tonlamalı (tek kanallı) rakam görüntülerinden oluşan bir veri setidir. Veriler Pad(2) ile 32x32 boyutuna getirilmiş ve Normalize edilmiştir.
- **CIFAR-10**: 32x32 boyutunda renkli (RGB, 3 kanallı) nesne görüntülerinden oluşan bir veri setidir. AlexNet ile uyum sağlayabilmesi için 224x224 boyutuna Resize işlemi uygulanmıştır.

### Model 1: Temel LeNet-5
- Yapı:
  - Conv2D(1,6,5x5) → ReLU → MaxPool
  - Conv2D(6,16,5x5) → ReLU → MaxPool
  - Conv2D(16,120,5x5) → ReLU
  - Fully Connected(120,84) → ReLU
  - Fully Connected(84,10)
- Optimizer: Adam (lr=0.01)
- Loss Fonksiyonu: CrossEntropyLoss

### Model 2: İyileştirilmiş LeNet-5 (BatchNorm + Dropout)
- Yapı:
  - Her Conv katmanından sonra Batch Normalization eklenmiştir.
  - Fully Connected katmanından sonra Dropout(p=0.5) eklenmiştir.
- Amaç: Öğrenmeyi hızlandırmak ve overfitting'i önlemek.
- Optimizer: Adam (lr=0.001)

### Model 3: Hazır AlexNet Mimarisi
- Model: torchvision.models.alexnet(weights=None)
- Çıkış katmanı (classifier[6]) 4096'dan 10'a dönüştürülmüştür.
- CIFAR-10 verisi 224x224'e resize edilmiştir.
- Optimizer: Adam (lr=0.001)

---

## Sonuçlar

| Model | Veri Seti | Epoch | Test Doğruluğu (%) |
|:---|:---|:---|:---|
| Model 1 (LeNet5) | MNIST | 10 | 98.22 |
| Model 2 (LeNet5V2) | MNIST | 10 | 99.08 |
| Model 3 (AlexNet) | CIFAR-10 | 5 | 63.96 |

### Loss Grafikleri (Epoch Başına Loss)
- Model 1: Loss, 0.2096'dan 0.0802'ye düşmüştür.
- Model 2: Loss, 0.2030'dan 0.0223'e düşmüştür.
- Model 3: Loss, 1.7241'den 1.0887'ye düşmüştür.

## Tartışma

- Model 1 başarılı sonuçlar vermiş, ancak BatchNorm ve Dropout gibi teknikler kullanılmadığı için overfitting riski taşımaktadır.
- Model 2'de eklenen BatchNorm ve Dropout ile çok daha stabil bir öğrenme sağlanmış ve test doğruluğu %99'un üzerine çıkmıştır.
- Model 3, CIFAR-10 gibi daha karmaşık bir veri seti kullanıldığından ve sadece 5 epoch eğitildiğinden, diğer modellere göre daha düşük bir başarım göstermiştir.
- AlexNet'in daha uzun eğitimi ve uygun öğrenme oranı ayarlamaları ile doğruluğu artırılabilir.

---

## Referanslar

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, 1998.
- A. Krizhevsky, I. Sutskever, G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," Advances in Neural Information Processing Systems (NeurIPS), 2012.
- PyTorch Resmi Dokümentasyonu: https://pytorch.org/docs/stable/index.html
- torchvision.models Dokümentasyonu: https://pytorch.org/vision/stable/models.html
- MNIST Veri Seti: http://yann.lecun.com/exdb/mnist/
- CIFAR-10 Veri Seti: https://www.cs.toronto.edu/~kriz/cifar.html

---

> Bu proje YZM304 Derin Öğrenme Dersi 2024-2025 Bahar Dönemi II. Ödevi kapsamında gerçekleştirilmiştir.

