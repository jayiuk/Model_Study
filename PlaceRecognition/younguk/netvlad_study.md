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
- 이 부분은 수정 해야할듯.