#### **CPU mode**

It is possible to run from script for all stages using the `--cpu-only` flag. To run from script, install the separate dependencies for CPU mode using `pip -r requirements-cpu.txt`.

Please note that extraction and training will take much long without a GPU and performance will greatly suffer without one. In particular, do not use DLIB extractor in CPU mode, it's too slow to run without a GPU. Train only on 64px resolution models like H64 or SAE (with low settings) and the lightweight encoder.