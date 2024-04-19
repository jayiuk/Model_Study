# NetVlad 코드 공부
## dataset
- natsorted를 사용해서 자연정렬

## NetVlad

### NetVlad
- base_model이 추출한 이미지 특성을 임베딩 하는 모델
- base_model과 같이 뒤의 EmbedNet의 생성 인자가 됨
- 합성곱 필터의 가중치는 self.centroids에 self.alpha를 곱한 값의 2배를 가중치로 설정
- 합성곱 필터의 편향은 self.centroids의 L2norm에 -self.alpha를 곱함
- 주어진 입력에 대해 순전파 연산을 수행함
- 입력 특징에 합성곱 연산 수행 -> 클러스터링된 특징
- 소프트맥수 함수 적용 -> 특징이 클러스터에 속할 확률
- 클러스터 중심과 입력 특징 간의 잔차 계산
- 잔차에 soft_assign 결과를 곱함 -> 각 클러스터에 대한 잔차를 가중치로 적용
- 가중치가 적용된 잔차를 각 클러스터별로 합산 -> vlad 벡터
- vlad벡터에 대해 L2정규화 수행
- vlad벡터를 2차원으로 펼친 후 다시 L2 정규화 -> 최종 전역 표현

### EmbedNet
- base_model과 net_vlad 두 개를 인자로 받음
- 특징 추출은 base_model, 임베딩 추출은 net_vlad를 사용
- 임베딩된 값을 반환

### TripletNet
- 여기선 앵커, 양성, 음성 이미지에 대한 임베딩 값을 생성함.
- 결국 앞에 모델들은 여길 위해서 사용되는 거라고 생각함

## train
- base_model로는 VGG16을 가져옴
  - 가져올 땐 마지막 layer는 잘라냄
  -   ```python
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg_layers = list(vgg_model.features.children())[:-1]
        model = nn.Sequential(*vgg_layers)
      ```
  - vgg 모델의 특성 추출기에서 자식 레이어인 합성곱 레이어와 풀링 레이어를 가져옴
  - base_model의 역할은 특성을 추출하는 것이기 때문
  - 그 다음 이를 리스트로 바꿔 마지막 부분을 제외
  - nn.Sequential을 통해 모델 생성. vgg_layers는 리스트 형태이므로 리스트 언팩킹을 해서 nn.Sequential에 사용
- criterion : tripletmarginloss를 사용함. 앵커, 양성, 음성 데이터 간의 거리를 최적화. 앵커와 양성은 가깝게, 음성은 더 멀게.
  - 여기선 margin을 0.1로 설정. 이는 앵커와 양성간의 거리 - 앵커와 음성간의 거리가 0.1이상은 되어야 한다는 조건
  - p=2는 거리 계산에 사용하는 norm의 차수. 여기선 유클리드 거리 방식을 사용하여 2로 설정.

- vlad_train
  - optimizer는 Adam을 사용
  - train_losses, test_losses는 train과 test시 나오는 loss값들을 저장하기 위한 리스트
  - 여기선 에포크마다 모델을 저장해주는 기능 구현
  - return은 train_losses, test_losses. 아마 리스트 형태로 나올듯. 일단 결과를 보고 굳이 리스트 형태로 볼 필요가 없으니 수정할 예정

- train_epochs
  - 실질적인 train을 전담
  - 매개변수로 optimizer를 사용
  - tqdm을 사용하여 progress bar 구현
    - total = len(train_loader)를 통해 프로그레스 바에 전체 반복 횟수 표현
    - enumerate(train_loader) : 데이터로데 객체에 대해 인덱스와 데이터를 함께 반복
  - running_loss += loss.item() -> 현재까지의 모든 배치에 대한 손실값을 스칼라로 반환
  - 마지막에 평균을 내서 리턴

- test_epochs
  - 모델을 평가
    - triplet.eval()
    - 평가를 하기 때문에 그래디언트를 계산하지 않음