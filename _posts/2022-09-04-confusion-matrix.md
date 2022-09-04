---
layout: post
title:  "CAM을 이용해서 semantic segmentation task를 해결하고 평가하는 방법"
date:   2022-09-04
categories: segmentation
description: "CAM을 이용해서 semantic segmentation task를 해결하고 평가하는 방법"
image: '/img/20220904/mIOU.png'
published: true 
---

오늘 소개해드릴 것은 Semantic Segmentation의 평가방식인 mIoU에 대해서 코드와 함께 분석해보려 한다.

또한, IRN [1],[2]을 참고하여 weakly supervised learning으로 어떻게 Semantic Segmentation task를 수행하는지 알아본다.

# mIOU

![image](/img/20220904/mIOU.png)

먼저 mIoU란 `Mean intersection over union`이라는 의미로, mean은 class마다 점수를 구하여 평균을 구하였다는 의미이고, Intersection over union은 전체 면적대비 맞은 pixel의 수를 의미한다. 즉 식은 아래와 같다.

```
IoU : (gt_pixel and predict_pixel) / (gt_pixel or predict_pixel)
```

즉 predict 된 곳이 ground truth와 얼마나 일치하는 지를 보여준다. 이 IoU를 class 별로 구하기 위해서 matrix 형태로 되어 있는 predict mask 와 gt mask를 비교한다. 쉽게 계산하기 위해서 두 matrix 모두를 1-dimension으로 `flatten`시키면 predicted vector와 gt vector가 1대1로 대응하게 된다. 그러면 둘을 비교해야 하는데, 이를 계산하기 위해서 여기서는 category matrix라는 것을 이용한다.

category matrix는 다음과 같은 식으로 각 pixel에 대응하는 category value들을 찾아낸다.

$category value = (gt value * class count) + predicted value$

이런 방식으로 mask크기 만큼의 1d vector의 값을 모두 구한다. 예를 들어, (1) 픽셀 하나가 predicted 한 class가 1이고, gt는 2이며, PASCAL VOC dataset (class가 20) 일때, category value는 2 * 20 + 1 = 41이다. (2) 두번째 예시로 predicted class가 5인데, gt class도 5라면 5 * 20 + 5 = 105 이다.

이 만들어진 category matrix를 이용하여 confusion matrix를 만든다. 만약 PASCAL VOC의 class 개수는 20이므로 confusion matrix의 크기는 20*20이 된다.

> 20*20 이 되는 이유는 matrix의 diagonal한 부분이 intersection, 그 이외의 부분은 모두 union이기 때문이다. 행렬의 coordinate에서 (1, 1)의 값 / {sum(1, i) + sum(j, 1)} 이 바로 class 1의 IoU라고 볼 수 있는 것이다.

예시 (1)에서 category value 를 41이라고 했는데, 이를 20으로 나누면 다시 quotient가 2, remainder가 1이므로 (2, 1) 좌표에 1을 더해준다. 그러면 (2)에서는 quotient가 5, remainder가 5로 predicted class와 gt class가 맞아, diagonal part 중 하나인 (5, 5)에 1이 더해지게 된다.

이는 np.bincount로 그 빈도수를 구하여 쉽게 계산이 가능하다. 이런식으로 모든 image에 대해서 confusion matrix가 값을 쌓은 후에, 각 class 마다 IoU를 구하고 이를 평균내어 mIOU 값을 구한다.

여기까지의 과정을 코드로 본다면 다음과 같다.   
`chainercv의 calc_semantic_segmentation_confusion 함수`
```python
...
pred_label = pred_label.flatten()
gt_label = gt_label.flatten()

# Dynamically expand the confusion matrix if necessary.
lb_max = np.max((pred_label, gt_label))
if lb_max >= n_class:
    expanded_confusion = np.zeros(
        (lb_max + 1, lb_max + 1), dtype=np.int64)
    expanded_confusion[0:n_class, 0:n_class] = confusion

    n_class = lb_max + 1
    confusion = expanded_confusion

# Count statistics from valid pixels.
mask = gt_label >= 0
confusion += np.bincount(
    n_class * gt_label[mask].astype(int) +
    pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))
...
```

보면 flatten하는 과정은 같은데, lb_max가 있습니다. 이는 class의 개수가 몇개인지 파악하기 위해서, mask에서 나온 가장 큰 class number를 가져와서 그 사이즈 만큼 confusion matrix를 늘려주는 역할을 한다. 만약 다른 이미지를 평가했는데, 거기서 이전에 나온 가장 큰 class number보다 더 큰 class number가 있으면 matrix를 한칸 더 확장합니다. `expanded_confusion`이 확장된 matrix의 예시다.

> 이런 방식으로 메모리를 너무 많이 잡아먹지 않도록 코드에서 조절해 준다.

그런 후에 `n_class * gt_label[mask].astype(int) + pred_label[mask], minlength=n_class**2)`로 바로 category matrix를 만들어주고, np.bincount를 이용해 confusion matrix를 update 해준다.

# CAM에서 predicted mask 뽑기.

[1]에서는 기존의 WSOL에서 thresholding 하는 방식인 `cv2.threshold`라는 함수를 쓰지 않고 다음과 같은 코드 flow를 이용한다.

```python
cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
cls_labels = np.argmax(cams, axis=0)
cls_labels = keys[cls_labels]
```

여기서 np.pad의 (1,0),(0,0),(0,0)은 즉, 첫번째 차원만 한칸 더 늘려주고 한칸 더 늘린 차원의 value는 `args.cam_eval_thres`로 설정해달라는 의미이다. 여기서 `args.cam_eval_thres`는 보통 0.1-0.5사이의 값을 지정해준다. (보통 0.1-0.15)  

한 차원을 늘렸다는 말은 현재 cams가 gt class label 만큼의 cam을 가지고 있는데, 만약 `cams`가 (2, 480, 300) 과 같은 차원을 가졌다고 치면 실제 ground truth class label이 두 개 있다는 의미이고, np.pad는 이를 (3, 480, 300)으로 늘려주고 한 차원은 모두 threshold값으로 초기화를 해준다는 말이다.  

그리고 이 `cams`의 1차원에 np.argmax 를 해준다. 즉 mask의 한 pixel에서 threshold보다 두 label의 cam의 모든 값이 낮으면 stuff로 처리하고, 그 이상이면 둘 중 더 큰 값을 가진 cam이 선택되게 되는 것. 이런 식으로 background class를 하나 더 만들어서 semantic segmentation mask를 생성한다.


Reference  
[1] Ahn et al., Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR, 2019  
[2] https://github.com/jiwoon-ahn/irn  
[3] mIoU(Mean Intersection over Union) 계산, https://gaussian37.github.io/vision-segmentation-miou/  