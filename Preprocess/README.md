# Preprocessing Data

English.

1. Before Pre-Processing, make sure prepare your dataset in local. </br>
   You can download the dataset in ai-hub, and The items used in this study are: </br>

- 013/01/1.Training/label_data/TL01/26 (Peripheral Neuropathy)
- 013/01/1.Training/audio_data/TS01/26 (Peripheral Neuropathy)
- 013/01/2.Validation/label_data/VL01/13 (Cerebral Palsy)
- 013/01/2.Validation/audio_data/VS01/13 (Cerebral Palsy)
- 013/01/2.Validation/label_data/VL01/11 (Stroke)
- 013/01/2.Validation/audio_data/VS01/11 (Stroke)

</br>

2. The `BASE_PATH` need to be edit to fit your dataset root.

3. If the number of preprocessed files does not match, process them manually.

4. If you want to use the file that has been preprocessed, go to the link below and use it. </br>

- [Stroke Dataset(뇌졸중)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Stroke_Dataset)
- [Cerebral Palsy Dataset(뇌성마비)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Celebral_Palsy_Dataset_Aug)
- [Peripheral Neuropathy Dataset(말초성 뇌신경장애)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset)
- [General Dataset(전체 학습)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset)

However, this dataset has been preprocessed to log-mel Spectogram. </br>
You can use it right away in the Whisper environment.

---

Korean.

1. 전처리를 하기 전에 데이터셋을 본인 환경에 다운받아야 합니다. </br>
   데이터셋은 ai-hub에서 다운받을 수 있으며, 사용한 데이터셋 경로는 아래와 같습니다: </br>

- 013/01/1.Training/label_data/TL01/26 (말초성 뇌신경장애)
- 013/01/1.Training/audio_data/TS01/26 (말초성 뇌신경장애)
- 013/01/2.Validation/label_data/VL01/13 (뇌성마비)
- 013/01/2.Validation/audio_data/VS01/13 (뇌성마비)
- 013/01/2.Validation/label_data/VL01/11 (뇌졸중)
- 013/01/2.Validation/audio_data/VS01/11 (뇌졸중)

2. 코드 내 `BASE_PATH`는 본인의 데이터셋이 저장된 경로로 수정되어야 합니다.

3. 과정 이후에도 전처리된 파일 간 개수가 일치하지 않으면, 수작업으로 처리해야합니다.

4. 만약 전처리가 다 완료된 파일을 바로 사용할 거라면, 아래 링크로 들어가서 사용하세요. </br>

- [Stroke Dataset(뇌졸중)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Stroke_Dataset)
- [Cerebral Palsy Dataset(뇌성마비)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Celebral_Palsy_Dataset_Aug)
- [Peripheral Neuropathy Dataset(말초성 뇌신경장애)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset)
- [General Dataset(전체 학습)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset)

단, 이 데이터셋은 log-mel Spectogram 전처리가 되어있습니다. Whisper 환경에서 바로 사용할 수 있습니다.
