# CMU Panoptic Datset

We present the detail of the Sequences and Cameras IDs on CMU Panoptic Datset.

## Sequences

Following [VoxelPose](https://arxiv.org/pdf/2004.06239), we use the following sequences:

**Training**

* 160422_ultimatum1
* 160224_haggling1
* 160226_haggling1
* 161202_haggling1
* 160906_ian1
* 160906_ian2
* 160906_ian3
* 160906_band1
* 160906_band2
* 160906_band3

**Inference**

* 160906_pizza1
* 160422_haggling1
* 160906_ian5
* 160906_band4

## Camera IDs

The camera IDs depend on the camera arrangements. Please see our paper for the detail description. 

| Camera Arrangements | Camera IDs | Camera Num |
| --- | --- | --- |
| CMU1 | 1, 2, 3, 4, 6, 7, 10 | 7 |
| CMU2 | 12, 16, 18, 19, 22, 23, 30 | 7 |
| CMU3 | 10, 12, 16, 18 | 4 |
| CMU4 | 6, 7, 10, 12, 16, 18, 19, 22, 23, 30 | 10 |
| CMU0 | 3, 6, 12, 13, 23 | 5 |
| CMU0 w/ 2 extra cameras | 3, 6, 12, 13, 23, 10, 16 | 7 |
| CMU0ex(K) | First K cameras in CMU0 w/ 2 | K |

