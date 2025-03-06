# MCUCoder: Adaptive Video Compression for IoT Edge Devices (Workshop on Machine Learning and Compression, NeurIPS 2024)

## Overview

MCUCoder is an open-source adaptive bitrate video compression model designed specifically for resource-constrained Internet of Things (IoT) devices. With a lightweight encoder requiring only 10.5K parameters and a memory footprint of 350KB, MCUCoder provides efficient video compression without exceeding the capabilities of low-power microcontrollers (MCUs) and edge devices.

## Features

- **Ultra-Lightweight Encoder**: Only 10.5K parameters, enabling efficient processing on MCUs.
- **Low Memory Usage**: 350KB memory footprint, making it ideal for edge devices with limited RAM (1-2MB).
- **High Compression Efficiency**: Reduces bitrate by **55.65% (MCL-JCV dataset)** and **55.59% (UVG dataset)** while maintaining visual quality.
- **Adaptive Bitrate Streaming**: Latent representation sorted by importance allows for dynamic transmission based on available bandwidth.
- **Comparable Energy Consumption to M-JPEG**: Ensures efficient power usage for real-time streaming applications.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

```
@inproceedings{
hojjat2024mcucoder,
title={{MCUC}oder: Adaptive Bitrate Learned Video Compression for IoT Devices},
author={Ali Hojjat and Janek Haberer and Olaf Landsiedel},
booktitle={Workshop on Machine Learning and Compression, NeurIPS 2024},
year={2024},
url={https://openreview.net/forum?id=ESjy0fQJJE}
}
```