# Install Guidance

**1. Environment Installation.** This part is tricky and take some efforts. Please make sure all components are installed correctly, no errors come out in each step, including the attention modules. 

When installing torch, please check your CUDA version using `nvcc --version`. And install pytorch with the correct versions. I tested with `CUDA 11.3` and `torch 1.12.1` this time.
```
conda create -n mvgformer python==3.10
conda activate mvgformer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -U openmim
mim install mmcv-full

pip install -r requirements.txt

cd ./lib/models/ops
CUDA_HOME=/usr/local/cuda-11.0/ python setup.py build install
```

When installing the attention modules, make sure that you use the correct cuda version with `CUDA_HOME`. Otherwise, there will be torch-CUDA mismatching error when compiling the attention module. Some more suggestions:
* If your torch version matches the cuda version with `nvcc --version`, then you do not need to set `CUDA_HOME` and it will automatically uses this default one.
* If you install another cuda-toolkit version in your conda environment, then you may need to set `CUDA_HOME=$CONDA_PREFIX`. But please double check according to your own system situation. 


**2. Dataset downloading.**
Please make sure if you downloaded the correct validation dataset, including the correct camera number, sequence names, extrinsics and instrinsics following the [guidance](https://github.com/XunshanMan/MVGFormer/blob/master/docs/CMU_sequences.md).

For convenience and sanity check, **I provide a new script in `scripts/getData_val_CMU0.sh` to download and postprocess all the CMU0 split** for you to run the validation command above. It already specifies correct camera number and comments unnecessary files. 

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
```

Copy the file `scripts/getData_val_CMU0.sh` to `${PANOPTIC_TOOLBOX_REPO}/scripts/`.

Repeat the following commands for 160906_pizza1, 160422_haggling1, 160906_ian5, 160906_band4.
```
cd ${PANOPTIC_TOOLBOX_REPO}
./scripts/getData_val_CMU0.sh 160906_pizza1 5 5
./scripts/extractAll.sh 160906_pizza1
```


3. Check the file organizations:

```
MVGFormer
  - data/panoptic/
      - 160422_haggling1
      - 160906_band4
      - 160906_ian5
      - 160906_pizza1
  - models/
      - mvgformer_q1024_model.pth.tar
      - pose_resnet50_panoptic.pth.tar 
  - run/validate_3d.py
  - README.md
```

4. Finally, run the code.

```
python run/validate_3d.py --cfg configs/panoptic/knn5-lr4-q1024.yaml --model_path models/mvgformer_q1024_model.pth.tar TEST.BATCH_SIZE=1
```

**Please make sure the data size is correct, e.g., you should get 2580 in the output:**

> Test: [100/**2580**]        Time: ...
