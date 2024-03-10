# Deep-Cardiac-Volumetric-Mesh (DeepCarve)

1. Heart mesh generation for biomechanics in <1 second
2. dcvm python package for pytorch model training & inference
    - training code coming soon
3. 3D Slicer visualization for inference

### Sample results (TMI 2023)

<img src="https://github.com/danpak94/Deep-Cardiac-Volumetric-Mesh/assets/21160618/b4d23754-e976-46bd-815b-45257cffe6e7" width="550">

<img src="https://github.com/danpak94/Deep-Cardiac-Volumetric-Mesh/assets/21160618/eed2dd5e-34e6-4d76-b1d5-d2251d34789e" width="900">

### System requirements

1. Windows or Linux (requires CUDA for now)

### Setup: 3D Slicer

1. Download the repository
    ```shell
    $ git clone https://github.com/danpak94/Deep-Cardiac-Volumetric-Mesh.git
    ```

2. [Download 3D slicer](https://download.slicer.org/)

3. Add SlicerDeepCardiac module(s)
\
\
**NOTE**: this will slicer.util.pip_install required packages
    - Select "Extension Wizard" module
        - Modules &rarr; Developer Tools &rarr; Extension Wizard
        - Or use "Find module" (magnifying glass next to "Modules:") and type "Extension Wizard"
    - Choose "Select Extension"
    - Select the "Deep-Cardiac-Volumetric-Mesh/SlicerDeepCardiac" folder
    - Follow prompts (check everything & say yes)

4. "ROI pred" module should be ready! (icon: <img src="SlicerDeepCardiac/RoiPred/Resources/Icons/RoiPred.png" width="20">)
    - Modules &rarr; SlicerDeepCardiac &rarr; ROI pred

### Quick start: ROI pred

(Only once during setup) "Setup &rarr; Download pre-trained model weights & mesh templates"
1. Load and select image volume
2. Define ROI
    - Only change "ROI center" for compatibility with dcvm pre-trained models
3. Load model weights and mesh template
4. "Crop & Run loaded model(s)"
5. "Save buttons &rarr; Save input node's outputs as .inp"

### Citation
```
@article{pak2023patient,
  title={Patient-specific Heart Geometry Modeling for Solid Biomechanics using Deep Learning},
  author={Pak, Daniel H and Liu, Minliang and Kim, Theodore and Liang, Liang and Caballero, Andres and Onofrey, John and Ahn, Shawn S and Xu, Yilin and McKay, Raymond and Sun, Wei and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
@inproceedings{pak2021distortion,
  title={Distortion Energy for Deep Learning-Based Volumetric Finite Element Mesh Generation for Aortic Valves},
  author={Pak, Daniel H and Liu, Minliang and Kim, Theodore and Liang, Liang and McKay, Raymond and Sun, Wei and Duncan, James S},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={485--494},
  year={2021},
  organization={Springer}
}
```
