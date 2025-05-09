# 🍎 Fruit Classifier

A PyTorch Image Classification model that uses a ResNet50 pretrained backbone to classify fruits in images.

---

## 📦 Project Structure

```

.
├── model.py               # YOLOv1-style model definition
├── train.py               # Training script
├── infer.py               # Inference script
├── class_to_idx.json      # Mapping of class names to indices
├── classes.json           # List of class names
├── checkpoints/           # Saved model weights & TensorBoard logs
├── requirements.txt
├── LICENSE
└── README.md

````

---

## 🛠️ Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. **Prepare dataset**

The dataset class will automatically download and split the dataset.

---

## 🚀 Training

Run the training script:

```bash
python train.py --epochs 50 --batch_size 32 --lr 1e-3 --device cuda
```

Arguments:

* `--epochs`: Number of training epochs
* `--batch_size`: Batch size
* `--lr`: Learning rate
* `--device`: "cuda" or "cpu"

The script will:

* Train the model with a pretrained ResNet50 backbone
* Save checkpoints to `checkpoints/`
* Log metrics to TensorBoard

---

## 🔍 Inference

You can use the Flask app for a simple web interface, simply run
```bash
python app.py
```


Use the trained model to detect objects in an image:

```bash
python infer.py /path/to/image.jpg --checkpoint checkpoints/model_best.pth --device cpu
```

Make sure `class_to_idx.json` and `classes.json` exist in the root folder.

---

## 📊 TensorBoard

To visualize training metrics:

1. Open a terminal in your project directory.

2. Run:

   ```bash
   tensorboard --logdir checkpoints/tb_logs
   ```

3. In Colab:

   ```python
   %load_ext tensorboard
   %tensorboard --logdir checkpoints/tb_logs
   ```

---

## 🧠 Model Architecture

* **Backbone**: Pretrained ResNet50 (frozen or fine-tuned)
* **Head**: 
Onve FC layer with a Dropout 

---

## 📁 Files You Need

* `class_to_idx.json`: mapping from class names to indices.
* `classes.json`: list of all class names.
* `checkpoints/model_best.pth`: trained model checkpoint.

---

## 📄 License

MIT License. See [LICENSE](./LICENSE) for details.