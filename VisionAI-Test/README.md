# VisionAI-Test

A comprehensive vision AI testing framework for image processing pipelines.

## Project Structure

```
VisionAI-Test/
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
│
├── images/
│   ├── input/               # Input images for processing
│   ├── masks/               # Generated masks
│   └── output/              # Output images
│
├── models/                  # ONNX models
│   ├── mobilesam/           # MobileSAM model files
│   │   ├── encoder.onnx
│   │   ├── encoder.data
│   │   ├── decoder.onnx
│   │   ├── decoder.data
│   │   └── metadata.json
│   │
│   ├── lama/                # LaMa inpainting model
│   │   └── lama_fp32.onnx
│   │
│   └── realesrgan/           # Real-ESRGAN super-resolution model
│       ├── real_esrgan_general_x4v3.onnx
│       ├── real_esrgan_general_x4v3.data
│       └── metadata.json
│
├── outputs/                 # Processing outputs
│   ├── debug/               # Debug outputs
│   └── final/               # Final outputs
│
├── scripts/                 # Test scripts
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_mask.py
│   ├── test_lama.py
│   ├── test_esrgan.py
│   └── test_pipeline.py
│
└── vision/                  # Main vision processing package
    ├── __init__.py
    ├── constants.py
    ├── config.py
    ├── utils.py
    ├── base.py
    ├── preprocessing.py
    ├── postprocessing.py
    ├── mobilesam.py
    ├── mask.py
    ├── lama.py
    ├── realesrgan.py
    └── pipeline.py
```

## Features

- **MobileSAM**: Segment Anything Model for mobile devices
- **LaMa**: Resolution-robust large mask inpainting
- **Real-ESRGAN**: Real-world blind super-resolution
- **Pipeline**: End-to-end image processing pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run test scripts to verify functionality:

```bash
python scripts/test_pipeline.py
python scripts/test_encoder.py
python scripts/test_decoder.py
python scripts/test_mask.py
python scripts/test_lama.py
python scripts/test_esrgan.py
```

## License

MIT License
