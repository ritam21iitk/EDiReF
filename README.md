# EDiReF
Emotion Discovery and Reasoning its Flip in Conversation
All code and models can also be found in the links below:
[ERC + EFR](https://drive.google.com/drive/folders/1caVkW4nahhVq8UFcoxfXh8h1WLwD8roZ?usp=sharing)  
[SPCL_SimCSE](https://drive.google.com/drive/folders/1It9bYNx94ebsgI3Cc-zpd4OYh7wMvNKh?usp=sharing)  
[SPCL_EFR](https://drive.google.com/drive/folders/1UyLdZndFs9bTclxYQ0H7n_yuAZfEazLx?usp=sharing)  
[SPCL_HBert_GRU](https://drive.google.com/drive/folders/1rkJnSdMWwVhLtGPBqFYNSXluMmGchhKR)  

### Confusion Matrices
### EFR-TX MaSaC

|           | 0 | 1 |
|-----------|---|---|
|            | 896 | 585 |
|            | 211 | 209 |

### EFR-TX MaSaC Hypothesis

|           | 0 | 1 |
|-----------|---|---|
|            | 691 | 263 |
|            | 238 | 177 |

### EFR-TX MELD

|           | 0 | 1 |
|-----------|---|---|
|            | 834 | 658 |
|            | 202 | 283 |

### EFR-TX MELD Hypothesis

|           | 0 | 1 |
|-----------|---|---|
|            | 370 | 469 |
|            | 168 | 303 |

### ERC MMN Hinglish

|           | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----------|---|---|---|---|---|---|---|---|
|           | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
|           | 0 | 12 | 0 | 0 | 1 | 62 | 0 | 0 |
|           | 0 | 3 | 0 | 0 | 0 | 18 | 0 | 0 |
|           | 0 | 4 | 0 | 2 | 0 | 50 | 0 | 0 |
|           | 0 | 3 | 0 | 0 | 0 | 33 | 0 | 0 |
|           | 0 | 12 | 0 | 2 | 0 | 236 | 0 | 0 |
|           | 0 | 2 | 0 | 0 | 0 | 46 | 0 | 0 |
|           | 0 | 1 | 0 | 0 | 0 | 42 | 0 | 0 |

### SPCL_SimCSE (For ERC problem)

#### i. Direct Confusion matrix

|           | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----------|---|---|---|---|---|---|---|---|
|           | 484 | 0 | 7 | 20 | 87 | 12 | 7 | 16 |
|           | 5 | 4 | 4 | 4 | 1 | 0 | 0 | 3 |
|           | 38 | 0 | 9 | 7 | 12 | 4 | 0 | 4 |
|           | 68 | 4 | 2 | 17 | 15 | 4 | 7 | 1 |
|           | 104 | 1 | 3 | 2 | 99 | 11 | 3 | 5 |
|           | 68 | 0 | 9 | 12 | 8 | 18 | 6 | 5 |
|           | 65 | 1 | 1 | 6 | 9 | 3 | 2 | 1 |
|           | 29 | 0 | 0 | 3 | 2 | 1 | 3 | 28 |

#### ii. Cluster Confusion matrix

|           | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----------|---|---|---|---|---|---|---|---|
|           | 416 | 0 | 22 | 29 | 76 | 29 | 33 | 28 |
|           | 2 | 5 | 6 | 3 | 1 | 0 | 0 | 4 |
|           | 27 | 2 | 14 | 5 | 10 | 6 | 6 | 4 |
|           | 52 | 5 | 5 | 21 | 11 | 8 | 15 | 1 |
|           | 87 | 1 | 6 | 5 | 98 | 19 | 5 | 7 |
|           | 48 | 1 | 10 | 13 | 9 | 31 | 8 | 6 |
|           | 46 | 1 | 3 | 10 | 8 | 5 | 13 | 2 |
|           | 21 | 0 | 1 | 2 | 2 | 5 | 2 | 33 |

### For SPCL_EFR (MELD)

#### i. Direct Confusion matrix

|           | 0 | 1 |
|-----------|---|---|
|           | 3023 | 0 |
|           | 489 | 0 |

#### ii. Cluster Confusion matrix

|           | 0 | 1 |
|-----------|---|---|
|           | 2765 | 258 |
|           | 391 | 98 |
