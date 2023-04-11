# Deep-Cardiac-Volumetric-Mesh

1. Heart mesh generation for biomechanics in <1 second
2. dcvm python package for pytorch model training & inference
    - training code coming soon
3. 3D Slicer visualization for inference

### Sample results on test set images (coming soon in publication form)

![image](https://user-images.githubusercontent.com/21160618/229584166-ff870933-1009-4826-8ae4-0ebde1cd7e07.png)

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
@inproceedings{pak2021distortion,
  title={Distortion Energy for Deep Learning-Based Volumetric Finite Element Mesh Generation for Aortic Valves},
  author={Pak, Daniel H and Liu, Minliang and Kim, Theodore and Liang, Liang and McKay, Raymond and Sun, Wei and Duncan, James S},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={485--494},
  year={2021},
  organization={Springer}
}
```
