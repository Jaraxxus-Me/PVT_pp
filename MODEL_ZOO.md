# PVT++ Model Zoo

## Introduction

This file documents a collection of baselines trained with PVT++. All configurations for these baselines are located in the [`experiments`](experiments) directory. The tables below provide results about inference. Links to the trained models as well as their output are provided. All the results are obtained on the same Nvidia Jetson AGX Xavier platform.

## *Online* Visual Tracking

|     <sub>Model</sub>      | <sub>DTB70</br> (mAUC/mDP) </sub> | <sub>UAVDT</br> (mAUC/mDP) </sub> | <sub>UAV20L</br> (mAUC/mDP) </sub> | <sub>UAV123</br> (mAUC/mDP) </sub> |                        <sub>URL</sub>                        |
| :-----------------------: | :-------------------------------: | :-------------------------------: | :--------------------------------: | :--------------------------------: | :----------------------------------------------------------: |
|    <sub>RPN_mob</sub>     |      <sub>0.298\|0.392</sub>      |      <sub>0.494\|0.719</sub>      |      <sub>0.448\|0.619</sub>       |      <sub>0.472\|0.678</sub>       | <sub>[RPN_Mob](https://mega.nz/file/8VlQXBIQ#ZbEBQnpMbQLJPQ0KqpALeHCZvxvOzW6QjTxX3hfnXS0)</sub> |
| <sub>RPN_mob+Motion</sub> |      <sub>0.385\|0.523</sub>      |      <sub>0.529\|0.745</sub>      |      <sub>0.481\|0.647</sub>       |      <sub>0.537\|0.737</sub>       | <sub>[RPN_Mob_M](https://mega.nz/file/hFVklIpZ#0M1VJ7C1zmz4NrfwqWVuVMKRVjyEHedqaAVco2UkYX8)</sub> |
| <sub>RPN_mob+Visual</sub> |      <sub>0.352\|0.472</sub>      |      <sub>0.564\|0.799</sub>      |      <sub>0.488\|0.675</sub>       |      <sub>0.504\|0.703</sub>       | <sub>[RPN_Mob_V](https://mega.nz/file/NRdlTTDS#TAcQwgEJmHLghFxFmDCTOv0gu5z57Eo3iiCaw-dRREw)</sub> |
|   <sub>RPN_mob+MV</sub>   |      <sub>0.399\|0.536</sub>      |      <sub>0.576\|0.807</sub>      |      <sub>0.508\|0.697</sub>       |      <sub>0.537\|0.741</sub>       | <sub>[RPN_Mob_MV](https://mega.nz/file/EVFxSSYB#4TFSJoVELbztvhJX8xkDlqwldmJT6XucHBEy9nINdlM)</sub> |

We also provide the [Raw_results](https://mega.nz/file/tFd02RxC#98PDk3XDhcXo9sZ-seKP5aklT0xC8rvbcUm77xu1Cmo).
These files can also be found at [Google Drive](https://drive.google.com/file/d/1oZjoHGGXqKSC43yKTwn2zwxFQprDXp7L/view?usp=sharing).
