-*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:percent
    text_representation:
      extension: .py
      format_name: percent
      format_version: '1.2'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: vietai
    language: python
    name: vietai
---

%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": "slide"}}
# Data representation
or Feature representation


%% [markdown] {"slideshow": {"slide_type": "subslide"}, "cell_style": "center"}
Use **Tensor** to represent features

* Image:
    * *flattened* vector
    * 3D Tensor of size $H\times W\times K$

* Text:
    * one-hot vector
    * count-based vector e.g. TF-iDF
    * "dense" vector learnt from Text Embedding model e.g. Word2Vec, BERT

* ...

%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": "slide"}}
# Machine Learning Models

%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": "fragment"}}
**Hypothesis function** - hàm giả thuyết  $\hat{f}:\mathcal{X}\rightarrow\mathcal{Y}$ được tham số hóa bằng $\theta$ với giả sử
* $\hat{f}$ tuyến tính (theo $\theta$) -> **Linear models**
* $\hat{f}$ biểu diễn bởi *Mạng nơ-ron* -> **DNN models**
* ...

%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": ""}}
**Thuật toán học** - tìm các tham số $\theta$ *chưa biết* của $\hat{f}$ bằng cách
* *tối thiểu hóa* **Hàm mục tiêu** $J\left(\theta;\mathcal{D}_{\left\{X,Y\right\}}\right)$ bằng thuật toán *Stochastic Gradient Descent - SGD* - hoặc biến thể
* ...

%% [markdown] {"slideshow": {"slide_type": "fragment"}}
<center><i>Quan trọng nhất</i> vẫn cần <b>Biểu diễn dữ liệu</b> i.e. <b>đặc trưng <i>đủ tốt</i></b></center>

%% [markdown] {"slideshow": {"slide_type": "slide"}, "cell_style": "center"}
## Linear Models

%% [markdown] {"slideshow": {"slide_type": ""}, "cell_style": "split"}
Good features
![](media/linear_model_good.png)

%% [markdown] {"slideshow": {"slide_type": ""}, "cell_style": "split"}
Bad features
![](media/linear_model_bad.png)

%% [markdown] {"slideshow": {"slide_type": "slide"}, "cell_style": "center"}
## DNN Models

%% [markdown] {"slideshow": {"slide_type": ""}, "cell_style": "center"}
<center><i>Automatic</i> feature extraction / Feature engineering by <b>architecture engineering</b></center>

%% [markdown] {"slideshow": {"slide_type": ""}, "cell_style": "split"}
![](media/DNN_shallow.png)

%% [markdown] {"slideshow": {"slide_type": ""}, "cell_style": "split"}
![](media/DNN_deep.png)

%% [markdown] {"slideshow": {"slide_type": "subslide"}, "cell_style": "center"}
### Architecture building blocks


%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": "-"}}
1. Non-linear activation

%% [markdown] {"cell_style": "split"}
Sigmoid, Tanh, ReLU, Maxout, ...

%% [markdown] {"cell_style": "split"}
2. Layers

%% [markdown] {"cell_style": "split"}
* Fully-connected / Dense layers
* Convolutional layers
* Pooling layers

%% [markdown] {"cell_style": "split"}
3. Skip connection / Residual connection

%% [markdown] {"cell_style": "split"}
(vd. mô hình *ResNet* - Lec10)

%% [markdown] {"cell_style": "split"}
4. Gated mechanism

%% [markdown] {"cell_style": "split"}
(vd. *LSTM cells, GRU cells* - Lec13)

%% [markdown] {"cell_style": "split"}
5. Attention mechanism

%% [markdown] {"cell_style": "split"}
(vd. mô hình *Seq2Seq* - Lec14,15)

%% [markdown] {"slideshow": {"slide_type": "subslide"}}
### Những lưu ý khi huấn luyện DNN

%% [markdown]
* Khởi tạo tham số
    * Không khởi tạo các trọng số $w$ bằng cùng một hằng số
    * Nên sử dụng *He init* (2015)
* Có thể sử dụng BatchNorm layers
* Chống overfit:
    * Sử dụng kỹ thuật Dropout
    * Áp dụng transfer learning (nếu có thể) khi dữ liệu huấn luyện không nhiều
    * Tăng cường dữ liệu huấn luyện - Data augmentation - nếu có thể
    * Cross-validate trên 1+ tập development/validation set có đủ tính đại diện
* Chống gradient exploding:
    * clip giá trị gradient không vượt quá một ngưỡng giới hạn cho trước


%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": "slide"}}
# Các mô hình DNN ứng dụng trong một số bài toán Xử lý ảnh, Xử lý Ngôn ngữ tự nhiên

%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": "subslide"}}
## Các bài toán **"hiểu"** ảnh/văn bản

%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": ""}}
* *Image classifcation - "Trong ảnh có gì?"* vd. xe máy / ô tô / landmarks / ...(Assignment 1-3)
    * Mô hình *VGG, ResNet* (Lec10)

* *Object detection - "Đối tượng ở vị trí nào trong ảnh?"*
    * Mô hình *YOLO, R-CNN*

* *Image segmentation*
    * Mô hình *U-Net* - mô hình gồm 2 CNNs - downsampling và upsampling CNN


%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": ""}}
* *Text classification - "Bài báo này nói về chủ đề gì?"* vd. Thể thao / Chính trị / ...
    * các mô hình CNN / RNN many-to-one
* *Sentiment Analysis - "Đoạn review này có ý khen hay chê?"* (Assignment 4)
    * các mô hình (bi-directional) RNN many-to-one

%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": ""}}
<center><i>Các mô hình trên đều được học trực tiếp bằng discriminative approach




%% [markdown] {"cell_style": "center", "slideshow": {"slide_type": "subslide"}}
## Các bài toán **"sinh"** ảnh/văn bản

%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": ""}}
* *(Machine) Translation - "Dịch văn bản từ ngôn ngữ này sang ngôn ngữ khác*"
    * Mô hình Seq2Seq (Lec13-15) - là loại mô hình có 2 RNNs đóng vai trò Encoder và Decoder tương ứng

%% [markdown] {"cell_style": "split", "slideshow": {"slide_type": ""}}
* *Image Translation - "Sinh ảnh, dựa trên một ảnh hướng dẫn"*
    * Mô hình *U-Net* - 2 CNNs đóng vai trò Encoder và Decoder tương ứng, học bằng adversarial training

 ![](https://camo.githubusercontent.com/6f486f0501ce4eef6b6050a0acedee8664c718b8/68747470733a2f2f7068696c6c6970692e6769746875622e696f2f706978327069782f696d616765732f7465617365725f76332e706e67)
