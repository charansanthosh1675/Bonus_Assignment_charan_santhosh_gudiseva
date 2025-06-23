## 🎓 Student Information
- **Name**: Charan santhosh gudiseva
- **Student ID**: 700776700
- **Course**: Neural Network & Deep Learning

---

## 🧾 Assignment Overview

This assignment demonstrates two core tasks using state-of-the-art deep learning models:

---

### Task 1: Question Answering System with Hugging Face Transformers

Objective:  
Build a question answering system using Hugging Face's Transformers library to extract answers from text based on context.

Key Steps:
- Used the `pipeline` API for quick QA model setup.
- Evaluated with both the default model and `deepset/roberta-base-squad2`.
- Created a custom context to ask multiple questions.

Tools: `transformers`, `torch`

## Sample Output:
```json
{
  "answer": "Charles Babbage",
  "score": 0.87,
  "start": 0,
  "end": 16
}
```


#Task 2: Digit-Class Controlled Image Generation with Conditional GAN


Objective:
The goal of this project is to implement a **Conditional Generative Adversarial Network (cGAN)** that generates MNIST digit images based on a given class label (0–9). This helps demonstrate how **conditioning a GAN** on class labels allows for controlled generation of specific outputs.

---

## 🛠️ Key Features & Architecture

- Generator:
  - Accepts both a noise vector and a **digit label**.
  - Concatenates the label (after embedding) with the noise.
  - Outputs a 28x28 grayscale digit image.
  
- Discriminator:
  - Takes an image and a label.
  - Concatenates the label embedding with the image (flattened).
  - Outputs a probability indicating whether the image-label pair is real or fake.

- Conditional Input:
  - Used `nn.Embedding` for converting labels into vectors.
  - Labels are injected into both the Generator and Discriminator.

---

## 📈 Training Setup

- Dataset: MNIST (handwritten digits)
- Epochs: 50 (can be adjusted for better results)
- Batch Size: 64
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (learning rate = 0.0002)

---

## 📊 Results

After training:
- The Generator successfully produces digits corresponding to the given labels.
- A row of 10 generated digits (0–9) visually confirms correct class conditioning.
