# Diagnosis-Aware Multitask Fine-Tuning of Whisper for Dysarthric Speech Recognition

[[English]](#English) [[Korean]](#Korean)

---

**English**<a name="English"></a>

This repository is designed to ensure the reproducibility of the article. <br />
You can run the research results through the following steps. <br />
**Note**: This study was conducted with Korean speech data.

---

## Research Content

People with dysarthria have irregular speech patterns depending on their disease, which causes problems with the recognition accuracy of commonly used voice recognition technologies. However, existing studies only compare single categories of diseases or utilize integrated dysarthria data, and do not classify disease-specific patterns for research. Therefore, this study extracts fluency indices for three diseases (stroke, cerebral palsy, and peripheral nerve disorders) based on the Korean speech disorder corpus, classifies the diseases using the extracted indices, and compares the performance of the speech recognition model. The experimental results show that the classification accuracy is 99% when the fluency index is used (p < 0.05), and the classified disease-specific model shows a difference of up to 18.34%p and 1.06%p in terms of CER, and up to 10.12%p and 0.79%p in terms of speaker-specific index (PER) compared to the Whisper-small model and the full learning model, respectively. This suggests that utilizing a detailed model affects the improvement of the error rate, and that a speaker-based customized speech recognition system is necessary.

---

## Comparison of utterance samples

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) | ![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Fig 1, 2. Disease-specific speech Visualization of waveform+VAD and spectrograms</p>
</br>
You can hear the voice on this page.

---

## Reproducibility

<p align='center'><img src="https://github.com/user-attachments/assets/e1b4ea86-8ea5-4531-a6c4-64f2e338ec55" width="2300" height="1080"/></p>
<p align='center'>Fig 3. Overview of our approach</p>

Note: The raw voice data is preprocessed, and feature extraction is performed. The extracted features are used to classify diseases. If there is misclassified voice data, the data is passed to the General Model that is trained to extract the text.

</br>

---

### Research Code and Model Repository

#### Research code

- Preprocess
- Feature Extraction
- Classfication
- Fine-Tuning
- Predict

</br >

#### Fine-Tuned Model

As the STT model before fine-tuning, we used openAI's Whisper-small model. [Whisper](https://github.com/openai/whisper) </br >

- [Stroke](https://huggingface.co/yoona-J/ASR_Whisper_Stroke)
- [Cerebral Palsy](https://huggingface.co/yoona-J/ASR_Whisper_Celebral_Palsy_Aug)
- [Peripheral Neuropathy](https://huggingface.co/yoona-J/ASR_Whisper_Peripheral_Neuropathy)
- [General](https://huggingface.co/yoona-J/ASR_Whisper_Disease_General)

</br >
  
### Preprocessed Training Datasets
Due to AI-Hub's policy, we cannot distribute the original data, so we provide it preprocessed as a log-Mel spectrogram. </br>
The original data can be downloaded through the link below.
[AI-Hub, "Speech recognition data for dysarthria"](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=608)
</br>

- [Stroke Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Stroke_Dataset)
- [Cerebral Palsy Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Celebral_Palsy_Dataset_Aug)
- [Peripheral Neuropathy Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset)
- [General Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset)

</br >

### Utterance Testset

#### Predict Utterance Dataset

- [Text_Preprocess](https://drive.google.com/drive/folders/1nj2i7ATyR_r-g64zfu_cPJrdQniVXv36?usp=drive_link)
- [Audio_Preprocess](https://drive.google.com/drive/folders/1gtUJdO5jkNTziiJcdAayRF_uyqLbpMqF?usp=drive_link)

</br >

#### Fluency Anakysis Values

We provide feature extraction figures as a file, which also includes speaker information (age, sex, disease). </br>

</br >

### Frameworks and Libraries Used

- Local Environment (python 3.10.1, pandas 2.2.3, numpy 1.26.4, openpyxl 3.1.5, noisereduce 3.0.3, praat-parselmouth 0.4.5, librosa 0.11.0)
- Google Colab Environment (python 3.11.13, transformers 4.54.0, torch 2.6.0+cu124, numpy 2.0.2, datasets 3.6.0, evaluate 0.4.4, librosa 0.11.0, nlptutti 0.0.0.10, jiwer 4.0.0)

### Training and Experimental Setup

- Local (NVIDIA GeForce MX150 & Intel® UHD Graphics 620 GPU, Intel Core i7-8565U @ 1.80GHz (4 cores / 8 threads) CPU, 16GB Ram)
- Google Colab (NVIDIA Tesla T4 GPU, 15GB VRAM, CUDA 12.4, Driver 550.54.15)

---

If you have any questions regarding the research, please contact us at the email below. </br>

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>

---

---

**Korean**<a name="Korean"></a>

해당 레포지토리는 논문의 재현성을 위해 구현되었습니다. <br />
연구 결과는 다음 단계를 따라 구현해볼 수 있습니다. <br />
**참고**: 본 연구는 한국어 음성 데이터를 사용해 수행되었습니다.

---

## Research Content

구음 장애인은 질환별로 불규칙한 음성 패턴을 가져 일반적으로 사용되는 음성 인식 기술에 대해 인식 정확도가 떨어지는 문제가 발생한다. 그럼에도 기존 연구에서는 단일 범주의 질환에 대해서만 비교하거나 통합된 구음 장애 데이터를 활용할 뿐 질환별 패턴을 분류해 연구하고 있지 않다. 따라서 본 연구는 한국어 구음 장애 코퍼스를 기반으로 세 가지 질환(뇌졸중, 뇌성마비, 말초성 뇌신경장애)에 대해 유창성 지표를 추출하고, 추출된 지표로 질환을 분류해 음성 인식 모델의 성능을 화자 특화 평가 지표인 PER과 함께 비교한다. 실험 결과, 유창성 지표를 활용했을 때 99%의 분류 정확도를 보이며(p < 0.05), 분류된 질환별 모델은 Whisper-small 모델과 전체 학습 모델보다 각각 최대 CER 기준 18.34%p, 1.06%p, 화자 특화 지표(PER)로는 각각 최대 10.12%p, 0.79%p의 차이를 나타내고 있다. 이는 모델을 세분화하여 활용하는 것이 오류율 개선에 영향을 미치며, 화자 기반 맞춤형 음성 인식 시스템이 필요함을 시사한다.

---

## Comparison of utterance samples

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) | ![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Fig 1, 2. 질환 별 음성 데이터의 파형+VAD 및 스펙트로그램 시각화</p>
</br>
Figure의 음성 샘플을 아래 링크를 통해 들어보실 수 있습니다.

---

## Reproducibility

<p align='center'><img src="https://github.com/user-attachments/assets/e1b4ea86-8ea5-4531-a6c4-64f2e338ec55" width="2300" height="1080"/></p>
<p align='center'>Fig 3. 연구 진행 방식</p>

참고: 원천 음성 데이터를 전처리하고, 전처리 된 데이터를 활용해 발화 특징을 추출합니다. 이는 질환 분류에 사용되며, 분류된 음성 데이터는 질환 별로 튜닝한 음성 인식 모델에 전달됩니다. 잘못 분류된 경우엔 질환 구분 없이 전체 데이터를 활용해 튜닝한 모델로 전달됩니다.

</br>

---

### Research Code and Model Repository

#### Research code

- Preprocess(전처리)
- Feature Extraction(특징 추출)
- Classfication(질환 분류)
- Fine-Tuning(모델 튜닝)
- Predict(텍스트 추출)

</br >

#### Fine-Tuned Model

튜닝 전 STT 모델로는 OpenAI 사의 Whisper-small 모델을 사용했습니다. [Whisper](https://github.com/openai/whisper) </br >

- [Stroke(뇌졸중)](https://huggingface.co/yoona-J/ASR_Whisper_Stroke)
- [Cerebral Palsy(뇌성마비)](https://huggingface.co/yoona-J/ASR_Whisper_Celebral_Palsy_Aug)
- [Peripheral Neuropathy(말초성 뇌신경장애)](https://huggingface.co/yoona-J/ASR_Whisper_Peripheral_Neuropathy)
- [General(전체 학습)](https://huggingface.co/yoona-J/ASR_Whisper_Disease_General)

</br >
  
### Preprocessed Training Datasets
본 연구에 사용된 음성 데이터는 AI-Hub의 정책 상 원본 데이터를 배포할 수 없어, log-mel spectogram으로 전처리된 수치 데이터로 제공합니다. </br>
원본 데이터는 아래 링크를 통해 다운받을 수 있습니다.
[AI-Hub, "구음장애 음성인식 데이터"](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=608)
</br>

- [Stroke Dataset(뇌졸중)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Stroke_Dataset)
- [Cerebral Palsy Dataset(뇌성마비)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Celebral_Palsy_Dataset_Aug)
- [Peripheral Neuropathy Dataset(말초성 뇌신경장애)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset)
- [General Dataset(전체 학습)](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset)

### Utterance Testset

#### Predict Utterance Dataset

- [Text_Preprocess](https://drive.google.com/drive/folders/1nj2i7ATyR_r-g64zfu_cPJrdQniVXv36?usp=drive_link)
- [Audio_Preprocess](https://drive.google.com/drive/folders/1gtUJdO5jkNTziiJcdAayRF_uyqLbpMqF?usp=drive_link)

</br >

#### Fluency Anakysis Values

음성 데이터를 통해 추출된 유창성 지표 수치와 화자 정보(나이, 성별, 질환)을 포함해 파일로 제공합니다. </br>

### Frameworks and Libraries Used

- Local Environment (python 3.10.1, pandas 2.2.3, numpy 1.26.4, openpyxl 3.1.5, noisereduce 3.0.3, praat-parselmouth 0.4.5, librosa 0.11.0)
- Google Colab Environment (python 3.11.13, transformers 4.54.0, torch 2.6.0+cu124, numpy 2.0.2, datasets 3.6.0, evaluate 0.4.4, librosa 0.11.0, nlptutti 0.0.0.10, jiwer 4.0.0)

### Training and Experimental Setup

- Local (NVIDIA GeForce MX150 & Intel® UHD Graphics 620 GPU, Intel Core i7-8565U @ 1.80GHz (4 cores / 8 threads) CPU, 16GB Ram)
- Google Colab (NVIDIA Tesla T4 GPU, 15GB VRAM, CUDA 12.4, Driver 550.54.15)

---

연구와 관련해서 질문이 있으시다면 아래 메일로 연락주세요.

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>
