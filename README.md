# Utilizing Language-Image Pretraining for Efficient and Robust Bilingual Word Alignment

This repository accompanies the paper:

**Utilizing Language-Image Pretraining for Efficient and Robust Bilingual Word Alignment**<br/>
Tuan Dinh, Jy-yong Sohn, Shashank Rajput, Timothy Ossowski, Yifei Ming, Junjie Hu, Dimitris Papailiopoulos, Kangwook Lee <br/>
Findings of the Association for Computational Linguistics: EMNLP 2022<br/>
[Paper Link](https://aclanthology.org/2022.findings-emnlp.12/)￼

**Overview**: This work introduces WALIP (Word Alignment using Language-Image Pretraining), a novel unsupervised method for bilingual word alignment that leverages pretrained language-image models. By utilizing the shared image-text embedding space from models like CLIP, WALIP enhances the efficiency and robustness of unsupervised word translation (UWT), particularly in scenarios lacking parallel corpora.

**Key Contributions:**
* Language-Image Pretraining: Employs CLIP’s shared embedding space to bridge semantic gaps between languages.
* Image-Based Fingerprints: Introduces a method to compute similarity confidences between word pairs using visual representations.
* Robust Procrustes Alignment: Applies an iterative algorithm to refine the linear mapping between embedding spaces, improving alignment accuracy.
* Empirical Validation: Demonstrates state-of-the-art performance across various language pairs and embedding types, showcasing robustness to language dissimilarity and corpus variations.

---

**Citation**

If you find this work useful, please cite:

```bibtex
@inproceedings{dinh-etal-2022-utilizing,
  title = "Utilizing Language-Image Pretraining for Efficient and Robust Bilingual Word Alignment",
  author = "Dinh, Tuan and Sohn, Jy-yong and Rajput, Shashank and Ossowski, Timothy and Ming, Yifei and Hu, Junjie and Papailiopoulos, Dimitris and Lee, Kangwook",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
  month = dec,
  year = "2022",
  address = "Abu Dhabi, United Arab Emirates",
  publisher = "Association for Computational Linguistics",
  pages = "154--168",
  url = "https://aclanthology.org/2022.findings-emnlp.12/",
  doi = "10.18653/v1/2022.findings-emnlp.12"
}
```
