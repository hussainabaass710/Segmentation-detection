# Model Architecture Choice: PointNet++

## Overview

This project uses PointNet++  for semantic segmentation of 3D point clouds. PointNet++ is a hierarchical neural network architecture that extends the original PointNet by incorporating local neighborhood information through hierarchical feature learning.

## Why PointNet++?

### 1. Hierarchical Feature Learning

PointNet++ addresses a key limitation of the original PointNet: it lacks the ability to capture local structures and spatial relationships between nearby points. 

During exploratory data analysis (EDA), we observed some correlation between heights and labels** in the ODMSemantic3D dataset. PointNet++ is particularly well-suited for this because:

- Implicit height awareness: The hierarchical feature extraction naturally accounts for changes in height through the multi-scale representation
- Spatial grouping: Ball query grouping captures local geometric patterns that include height variations
- Multi-resolution processing: Different scales of the hierarchy can capture height-related features at various levels of detail

Our data preprocessing extracts fixed-size patches (4096 points each) using ball query, which aligns perfectly with PointNet++:

- Consistent input size: All patches have exactly 4096 points, matching the model's expected input
- Local feature learning: Each patch is processed independently, allowing the model to learn local patterns
- Efficient training: Patch-based training enables handling of large point clouds that wouldn't fit in memory otherwise


PointNet++ has demonstrated strong performance on:
- Semantic segmentation benchmarks
- Part segmentation tasks
- Scene understanding applications
- Point cloud classification

The architecture has been widely adopted and validated in the research community, making it a reliable choice for production applications.

