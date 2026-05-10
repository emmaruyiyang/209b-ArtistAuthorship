# Artist Authorship Attribution

50-class artist attribution from painting images. DS2 Group #166 (Yiqiao Huang, Lixuan Wei, Ruyi Yang, Tian Xia).

**Dataset:** [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data) — 50 artists, 8,355 images.


## Pipeline

![pipeline](fig/209b-project-pipeline.png)

| Model | Test Acc |
|---|---|
| ResNet-18 (full fine-tune) | **0.789** |
| ViT-B/16 (linear probe) | 0.648 |
| CLIP zero-shot | 0.638 |
| EfficientNet-B0 (linear probe) | 0.560 |



## Slides

[Milestone 2](https://docs.google.com/presentation/d/1Yi-8sNqPw_74J4dzwzK7wS77EnGcdfKkU6x6f4AiYfM/edit) ·

[Milestone 3](https://docs.google.com/presentation/d/1Ckgo10r7gyPSLdDYkWb_ZmAkyDMMTDYeBELNgo0ZGvg/edit)
