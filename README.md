# 手写数字识别系统

基于PyTorch和Flask构建的手写数字识别Web应用。

## 功能特性

- ✏️ **手写输入**: 使用鼠标或触摸屏在画板上书写数字
- 📤 **图片上传**: 支持点击选择和拖拽上传图片
- 🧠 **CNN模型**: 使用卷积神经网络进行数字识别
- 📊 **实时识别**: 显示识别结果和置信度

## 技术栈

- Python 3.x
- PyTorch
- Flask
- PIL/Pillow

## 项目结构

```
.
├── app.py                 # Flask Web应用
├── train_cnn.py           # CNN模型训练脚本
├── templates/
│   └── index.html         # 前端页面
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install torch torchvision flask pillow pandas numpy matplotlib
```

## 训练模型

```bash
python train_cnn.py
```

训练完成后会生成：
- `mnist_cnn.pth` - 训练好的模型权重
- `sample_submission.csv` - 测试集预测结果
- `loss_curve.png` - 训练损失曲线图

## 启动Web应用

```bash
python app.py
```

访问 http://localhost:7860 即可使用手写数字识别功能。

## 使用说明

1. **手写输入**: 选择"手写输入"标签，使用鼠标在画板上书写数字，点击"识别"按钮获取结果
2. **图片上传**: 选择"图片上传"标签，点击上传区域或拖拽图片，点击"识别"按钮获取结果
3. **清除**: 点击"清除"按钮清空画板或上传的图片

## 模型架构

```
输入 (1x28x28)
    ↓
Conv1 (32个3x3卷积核) → ReLU → MaxPool
    ↓
Conv2 (64个3x3卷积核) → ReLU → MaxPool
    ↓
Flatten (64×7×7)
    ↓
FC1 (128) → ReLU
    ↓
FC2 (10) → 输出
```

## 数据集

使用MNIST手写数字数据集，包含60000张训练图片和10000张测试图片。

## 许可证

MIT License