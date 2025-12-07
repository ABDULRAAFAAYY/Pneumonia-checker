# 🫁 Pneumonia Detection AI

An AI-powered Streamlit application that detects pneumonia from chest X-ray images using Hugging Face's deep learning model.

## Features

- **Upload chest X-ray images** (PNG, JPG, JPEG)
- **AI-powered analysis** using ResNet50V2 model
- **Instant results** with confidence scores
- **Premium dark theme UI** with modern design

## Model

Uses the [ryefoxlime/PneumoniaDetection](https://huggingface.co/ryefoxlime/PneumoniaDetection) model from Hugging Face, trained on chest X-ray images to classify Normal vs Pneumonia cases.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It should NOT be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
