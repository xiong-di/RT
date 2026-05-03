## Overview

Package Requirements: torch, torchaudio, timm, scikit-learn, numpy

### Preparing the data

You can prepare the data by following the instructions of the [benchmark](https://github.com/XLearning-SCU/2024-ICLR-READ).

### Running the code

You can run the code using the command:

```bash
python main.py --dataset 'ks50' --json-root [json-root] --label-csv [label-csv] --pretrain_path [pretrain_path] --tta-method 'rsa' --severity 55 --corruption-modality 'both'
```
