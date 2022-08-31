---
layout: post
title:  "Weakly-Supervised 에서의 Localization, Semantic, Instance, Panoptic Segmenation"
date:   2022-08-30
categories: segmentation
description: "Localization, Semantic, Instance, Panoptic Segmenation"
image: '/img/20220830/semantic-instance-panoptic.png'
published: true 
---

이번에 소개하려는 주제는 weakly supervised learning에서 주로 다루는 task인 **object localization, semantic segmentation**과, 조금 더 어려운 task인 instance, panoptic segmentation에 대해서 소개하겠습니다. (간단하게 소개하는 글입니다.)

![image](/img/20220830/semantic-instance-panoptic.png)

이 글은 자료 [1]을 많이 참고하였으며, object localization에 대한 정보는 weakly supervised object localization 논문들([2], [3])을 참고하였습니다.

먼저 핵심만 말하자면, task의 어려운 정도는 (주관적으로) 다음과 같이 볼 수 있습니다.  
> localization < semantic seg < instance seg < panoptic seg

먼저 object localization task는 각 이미지의 single object를 얼마나 잘 detection, segmentation 하는 지의 task입니다. 특히 evlauation 기준을 고려하였을 때 ImageNet과 CUB는 bounding box를 이용해서 평가하는데, CUB는 거의 모든 이미지가 single object만 annotation이 되어있으며, ImageNet은 multi object가 annotation 되어 있더라도 한 개의 object의 bounding box만 잘 나오면 localization이 잘 되었다고 평가합니다(MaxBoxAcc [3]). (자세한 평가방식은 나중에)

따라서, weakly supervised object localization의 경우, `minmax`로 class activation map (CAM)을 normalization 하던 0-1사이로 CAM이 생성되면 상관이 없습니다 [4]. 왜냐하면 single object만 제대로 localization 해서, bounding box를 찾으면 그만이기 때문입니다.

<img src="/img/20220830/loc_img.png" width="200px" height="100px"/>

그러나 semantic segmentation의 경우에는 다릅니다. 이미지내에는 Multiple instance, multiple class가 존재하고, 이를 모두 예측해야 합니다. 즉 제가 첨부하였던 맨 첫번째 이미지에서 오른쪽 위부분의 그림이 semantic segmentation입니다.

<img src="/img/20220830/sem_img.png" width="200px" height="100px"/>

결론적으로, 만약 weakly supervised object localization (WSOL)의 기술을 weakly supervised semantic segmentation에 적용하고 싶다면, `minmax`가 아닌 `max` normalization을 이용해야 하고, max normalization을 진행할 때도 negative 값을 그대로 살려야 합니다. semantic segmentation에서 셀 수 있는 물체 즉, **Thing**이 아닌 것들은 샐 수 없는 것들인 **stuff**로 분류해 낼 수 있어야 하는 중요한 점이 있습니다. 그렇기 때문에 negative 값을 그대로 유지해야 이 negative 값들을 stuff로 인지 할 수 있습니다. 위의 사진 중에서 빨간색으로 일정하게 존재하는 값들은 모두 stuff로 취급된 것들입니다. localization 과 다른 CAM이 생성된다는 것을 알 수 있습니다.

> 짚고 넘어가야 할 점은, 제가 보여드린 그림은 WSSS에서 semantic segmentation network를 학습시키기 전에 classification network로 부터 만든 pseudo label seed 입니다.

Instance segmentation에서는 위에서 말한 셀 수 있는 것, 즉 **Thing**을 구별해 낼 수 있어야 합니다. 물체마다 같은 class이더라도 다른 boundary를 정확하게 생성해내야 하는 task인데, class label만이 존재하는 weakly supervised instance segmentation은 성능이 기존에 비해 많이 낮기 때문에 보통 bounding box를 이용하거나 [6], pointly-supervised 기술[5]을 이용하여 성능을 fully supervised 의 성능을 높히려는 시도를 많이 합니다.

Panoptic segmentation은 semantic과 instance를 모두 합친 task입니다. 즉 stuff와 thing 모두를 인식해야하는 문제로, 가장 어렵습니다. 그래서 2018년도에 나온 [8]에서는 weakly and semi supervised learning을 통해서 fully supervised learning의 95%정도까지 따라잡았지만, 만약 Image Label만 이용하게 된다면 작년에 나온 논문인 [7]에서 report 한 바가 있듯이, fully supservised setting 보다 성능이 훨씬 낮다는 것을 알 수 있습니다.

> 성능이 낮아서 그런지 몰라도, qualitative result를 report 하지 않았습니다..

또한 비교 대상도 부족하기 때문에 Instance Segmentation을 같이 수행하기도 합니다. 해당 부분의 task는 image label단위의 weak supervision보다 scrubble이나 box, point와 같은 weak label을 이용해야 할 필요가 있을 것 같습니다.






Reference

[1] Semantic vs Instance vs Panoptic: Which Image Segmentation Technique To Choose, https://analyticsindiamag.com/semantic-vs-instance-vs-panoptic-which-image-segmentation-technique-to-choose/  
[2] Zhou et al, Learning Deep Features for Discriminative Localization, CVPR, 2016.  
[3] Choe et al, Evaluating Weakly Supervised Object Localization Methods Right, CVPR, 2020.  
[4] Kim et al, Normalization Matters in Weakly Supervised Object Localization, ICCV, 2021.  
[5] Cheng et al, Pointly-Supervised Instance Segmentation, CVPR, 2022.  
[6] Tian et al, BoxInst: High-Performance Instance Segmentation with Box Annotations, CVPR, 2021.  
[7] Shen et al, Toward Joint Thing-and-Stuff Mining for Weakly Supervised Panoptic Segmentation, CVPR, 2021.  
[8] Li et al, Weakly- and Semi-Supervised Panoptic Segmentation, ECCV, 2016.
