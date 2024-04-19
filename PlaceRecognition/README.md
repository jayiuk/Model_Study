실험 수행 방법
1. experiments 디렉토리에 실험디렉토리 생성 (ex: experiments/sample)
2. 실험디렉토리에 실험하고자 하는 파라미터가 담긴 setup.ini 파일 저장 (메인 디렉토리의 setup.ini 가 일차적으로 로드된 뒤, 해당 setup.ini 가 로드됨)
3. GPU 개수, 실험디렉토리명을 인자로 주어 학습 진행 (예: torchrun --nproc_per_node=4 train.py experiments/sample/)
