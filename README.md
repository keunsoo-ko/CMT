![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# CMT

Keunsoo Ko and Chang-Su Kim

Official PyTorch Code for "Continuously Masked Transformer for Image Inpainting, ICCV, 2023"

### Installation
Download repository:
```
    $ git clone https://github.com/keunsoo-ko/CMT.git
```
Download [pre-trained model, Places2](https://drive.google.com/file/d/1zLkKixPnuoAY1k4fdq6JjidlRafKMhXN/view?usp=sharing) or [pre-trained model, CelebA](https://drive.google.com/file/d/1e6EbwGnMGgGXAn4QLffT_Zx_BbidBSbR/view?usp=sharing)

### Usage
Run Test for the spatial super resolution on the HCI dataset with the factor x2:
```
    $ python demo.py --mode SR --path LFSR-AFR.pth(put downloaded model path)
```
Run Test for the angular super resolution on the HCI dataset with the factor x2:
```
    $ python demo.py --mode AR --path LFSR-AFR.pth
```
