# Text-Conditioned Object-Relative Geometry for Open-Vocabulary Part Segmentation

Computer Vision project investigating whether explicit object-relative geometry
provides useful and interpretable spatial information beyond pretrained visual
features for text-conditioned part segmentation.

## Main Components

- DINOv2 visual encoder
- CLIP text encoder
- Parent-object mask conditioning
- Absolute image coordinates (X, Y)
- Object-relative coordinates (U, V)
- Boundary-distance map (D)
- Query-conditioned geometry gating

## Planned Experiments

1. DINOv2 + text baseline
2. + parent-object mask
3. + absolute image coordinates (X, Y)
4. + object-relative coordinates (U, V)
5. + fixed object-relative geometry (U, V, D)
6. + query-gated object-relative geometry (U, V, D)
7. Rotation robustness
8. Noisy parent-mask robustness