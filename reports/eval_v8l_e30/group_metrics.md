| group | value | num_images | num_ped | mAP50 | mAP50-95 | recall |
|---|---|---|---|---|---|---|
| all | all | 10000 | 13262 | 0.706 | 0.3817 | 0.6149 |
| timeofday | dawn/dusk | 778 | 1046 | 0.6925 | 0.3718 | 0.6195 |
| timeofday | daytime | 5258 | 9366 | 0.7199 | 0.3931 | 0.6281 |
| timeofday | night | 3929 | 2843 | 0.664 | 0.3475 | 0.5867 |
| timeofday | undefined | 35 | 7 | 0.8995 | 0.4705 | 0.8571 |
| weather | clear | 5346 | 4935 | 0.6951 | 0.3735 | 0.6132 |
| weather | overcast | 1239 | 1916 | 0.7227 | 0.3968 | 0.6284 |
| weather | partly cloudy | 738 | 1034 | 0.6787 | 0.368 | 0.5698 |
| weather | rainy | 738 | 822 | 0.6611 | 0.3596 | 0.5782 |
| weather | snowy | 769 | 1137 | 0.7525 | 0.4126 | 0.672 |
| weather | undefined | 1157 | 3414 | 0.7182 | 0.3866 | 0.6292 |
| crowd | crowded | 351 | 5255 | 0.7143 | 0.3677 | 0.6084 |
| crowd | not_crowded | 9649 | 8007 | 0.7049 | 0.3943 | 0.6194 |
| size | small | 10000 | 6399 | 0.4756 |  | 0.7598 |
| size | medium | 10000 | 6108 | 0.8643 |  | 0.9635 |
| size | large | 10000 | 755 | 0.9517 |  | 0.9762 |

注: size 组的 mAP50 由 size_ap.py 自行计算 (全点插值), 与 Ultralytics 的数值不完全可比。
