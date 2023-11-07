# EDiReF
Emotion Discovery and Reasoning its Flip in Conversation
All code and models can also be found in the links below:
https://drive.google.com/drive/folders/1caVkW4nahhVq8UFcoxfXh8h1WLwD8roZ?usp=sharing
https://drive.google.com/drive/folders/1It9bYNx94ebsgI3Cc-zpd4OYh7wMvNKh?usp=sharing
https://drive.google.com/drive/folders/1UyLdZndFs9bTclxYQ0H7n_yuAZfEazLx?usp=sharing
https://drive.google.com/drive/folders/1rkJnSdMWwVhLtGPBqFYNSXluMmGchhKR  
### Confusion Matrices
### EFR-TX MaSaC

|           | Predicted: 0 | Predicted: 1 |
|-----------|--------------|--------------|
| Actual: 0 | 896          | 585          |
| Actual: 1 | 211          | 209          |

### EFR-TX MaSaC Hypothesis

|           | Predicted: 0 | Predicted: 1 |
|-----------|--------------|--------------|
| Actual: 0 | 691          | 263          |
| Actual: 1 | 238          | 177          |

### EFR-TX MELD

|           | Predicted: 0 | Predicted: 1 |
|-----------|--------------|--------------|
| Actual: 0 | 834          | 658          |
| Actual: 1 | 202          | 283          |

### EFR-TX MELD Hypothesis

|           | Predicted: 0 | Predicted: 1 |
|-----------|--------------|--------------|
| Actual: 0 | 370          | 469          |
| Actual: 1 | 168          | 303          |

### ERC MMN Hinglish

|           | Pred: 0 | Pred: 1 | Pred: 2 | Pred: 3 | Pred: 4 | Pred: 5 | Pred: 6 | Pred: 7 |
|-----------|---------|---------|---------|---------|---------|---------|---------|---------|
| Actual: 0 | 0       | 0       | 0       | 0       | 0       | 9       | 0       | 0       |
| Actual: 1 | 0       | 12      | 0       | 0       | 1       | 62      | 0       | 0       |
| Actual: 2 | 0       | 3       | 0       | 0       | 0       | 18      | 0       | 0       |
| Actual: 3 | 0       | 4       | 0       | 2       | 0       | 50      | 0       | 0       |
| Actual: 4 | 0       | 3       | 0       | 0       | 0       | 33      | 0       | 0       |
| Actual: 5 | 0       | 12      | 0       | 2       | 0       | 236     | 0       | 0       |
| Actual: 6 | 0       | 2       | 0       | 0       | 0       | 46      | 0       | 0       |
| Actual: 7 | 0       | 1       | 0       | 0       | 0       | 42      | 0       | 0       |

### SPCL_SimCSE (For ERC problem)

#### i. Direct Confusion matrix

|           | Pred: 0 | Pred: 1 | Pred: 2 | Pred: 3 | Pred: 4 | Pred: 5 | Pred: 6 | Pred: 7 |
|-----------|---------|---------|---------|---------|---------|---------|---------|---------|
| Actual: 0 | 484     | 0       | 7       | 20      | 87      | 12      | 7       | 16      |
| Actual: 1 | 5       | 4       | 4       | 4       | 1       | 0       | 0       | 3       |
| Actual: 2 | 38      | 0       | 9       | 7       | 12      | 4       | 0       | 4       |
| Actual: 3 | 68      | 4       | 2       | 17      | 15      | 4       | 7       | 1       |
| Actual: 4 | 104     | 1       | 3       | 2       | 99      | 11      | 3       | 5       |
| Actual: 5 | 68      | 0       | 9       | 12      | 8       | 18      | 6       | 5       |
| Actual: 6 | 65      | 1       | 1       | 6       | 9       | 3       | 2       | 1       |
| Actual: 7 | 29      | 0       | 0       | 3       | 2       | 1       | 3       | 28      |

#### ii. Cluster Confusion matrix

|           | Pred: 0 | Pred: 1 | Pred: 2 | Pred: 3 | Pred: 4 | Pred: 5 | Pred: 6 | Pred: 7 |
|-----------|---------|---------|---------|---------|---------|---------|---------|---------|
| Actual: 0 | 416     | 0       | 22      | 29      | 76      | 29      | 33      | 28      |
| Actual: 1 | 2       | 5       | 6       | 3       | 1       | 0       | 0       | 4       |
| Actual: 2 | 27      | 2       | 14      | 5       | 10      | 6       | 6       | 4       |
| Actual: 3 | 52      | 5       | 5       | 21      | 11      | 8       | 15      | 1       |
| Actual: 4 | 87      | 1       | 6       | 5       | 98      | 19      | 5       | 7       |
| Actual: 5 | 48      | 1       | 10      | 13      | 9       | 31      | 8       | 6       |
| Actual: 6 | 46      | 1       | 3       | 10      | 8       | 5       | 13      | 2       |
| Actual: 7 | 21      | 0       | 1       | 2       | 2       | 5       | 2       | 33      |

### For SPCL_EFR (MELD)

#### i. Direct Confusion matrix

|           | Pred: 0 | Pred: 1 |
|-----------|---------|---------|
| Actual: 0 | 3023    | 0       |
| Actual: 1 | 489     | 0       |

#### ii. Cluster Confusion matrix

|           | Pred: 0 | Pred: 1 |
|-----------|---------|---------|
| Actual: 0 | 2765    | 258     |
| Actual: 1 | 391     | 98      |

