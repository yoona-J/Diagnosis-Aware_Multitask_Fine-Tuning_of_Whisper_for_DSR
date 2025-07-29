# Diagnosis-Aware Multitask Fine-Tuning of Whisper for Dysarthric Speech Recognition

[[English]](#English) [[Korean]](#Korean)

---

**English**<a name="English"></a>

This repository is designed to ensure the reproducibility of the article. <br />
You can run the research results through the following steps. <br />
**Note**: This study was conducted with Korean speech data.

---

## Research Content

Individuals with dysarthria exhibit irregular speech patterns depending on the characteristics of their disease, significantly reducing the accuracy of conventional speech recognition systems. Most prior studies have compared only a single disease group or used aggregated data without distinguishing between diseases, failing to adequately analyze disease-specific differences. This study extracted fluency metrics from a Korean dysarthric speech corpus across three disease groups—stroke, cerebral palsy, and peripheral neuropathy—and classified the diseases based on these features. Then, the performance of customized speech recognition models for each disease was evaluated using Weighted Character Error Rate (Weighted-CER). The results showed that the classification accuracy based on fluency metrics reached 99%, and the disease-specific models improved Weighted-CER by up to 18.34 and 1.05 percentage points compared to the Whisper-Small model and a model trained on the entire dataset, respectively. In terms of Weighted-CER, the error rate decreased by up to 15.27 and 1.49 percentage points, respectively. These findings indicate that disease-specific models can meaningfully enhance speech recognition performance for dysarthric speech and highlight the necessity of developing speaker-customized speech recognition systems.

---

## Comparison of utterance samples

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) | ![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Fig 1, 2. Disease-specific speech Visualization of waveform+VAD and spectrograms</p>

---

## Reproducibility

<img width="2300" height="1080" alt="ASR_Figure4" src="https://github.com/user-attachments/assets/f7ee6291-e478-4fa2-a202-2ad1f95a8ca3" />
<p align='center'>Fig 3. Overview of our approach</p>

Note: The raw voice data is preprocessed, and feature extraction is performed. The extracted features are used to classify diseases. If there is misclassified voice data, the data is passed to the General Model that is trained to extract the text.

</br>

---

### Research Code and Model Repository

#### Research code

- [Preprocess](Preprocess)
- [Feature Extraction](Feature_Extraction)
- [Classfication](Classfication)
- [Fine-Tuning](Fine-Tuning)
- [Predict](Predict)

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

#### Fluency Analysis Values

We provide feature extraction figures as a file, which also includes speaker information (age, sex, disease). </br>

- [fluency_analysis](Feature_Extraction/Result/fluency_analysis.csv)

</br >

### Frameworks and Libraries Used

- Local Environment (python 3.10.1, pandas 2.2.3, numpy 1.26.4, openpyxl 3.1.5, noisereduce 3.0.3, praat-parselmouth 0.4.5, librosa 0.11.0)
- Google Colab Environment (python 3.11.13, transformers 4.54.0, torch 2.6.0+cu124, numpy 2.0.2, datasets 3.6.0, evaluate 0.4.4, librosa 0.11.0, nlptutti 0.0.0.10, jiwer 4.0.0)

</br >

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

구음 장애인은 질환 특성에 따라 불규칙한 음성 패턴을 보이며, 이로 인해 기존 음성 인식 시스템의 정확도가 크게 낮아지는 문제가 있다. 그러나 선행 연구 대부분은 단일 질환 집단만 비교하거나 질환을 구분하지 않은 통합 데이터를 사용해 질환별 차이를 충분히 분석하지 못했다. 본 연구는 한국어 구음 장애 코퍼스를 활용해 뇌졸중, 뇌성마비, 말초성 뇌신경 장애의 세 질환 그룹에서 유창성 지표를 추출한 뒤, 해당 지표로 질환을 분류하고 각 질환별 맞춤형 음성 인식 모델의 성능을 가중 문자 오류율(Weighted-CER)로 평가했다. 실험 결과, 유창성 지표 기반 분류 정확도는 99%에 달했으며, 질환별로 학습한 모델은 Whisper-small 모델 대비 최대 18.34%p, 전체 데이터를 학습한 모델 대비 최대 1.05%p의 CER 개선 효과를 보였다. Weighted-CER 기준으로는 각각 최대 15.27%p, 1.49%p까지 오류율이 감소했다. 이는 질환별로 세분화된 모델이 구음 장애 음성 인식 성능을 유의미하게 향상시킬 수 있음을 시사하며, 화자 맞춤형 음성 인식 시스템 개발의 필요성을 강조한다.

---

## Comparison of utterance samples

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) | ![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Fig 1, 2. 질환 별 음성 데이터의 파형+VAD 및 스펙트로그램 시각화</p>

---

## Reproducibility

<img width="2300" height="1080" alt="ASR_Figure4" src="https://github.com/user-attachments/assets/f7ee6291-e478-4fa2-a202-2ad1f95a8ca3" />
<p align='center'>Fig 3. 연구 진행 방식</p>

참고: 원천 음성 데이터를 전처리하고, 전처리 된 데이터를 활용해 발화 특징을 추출합니다. 이는 질환 분류에 사용되며, 분류된 음성 데이터는 질환 별로 튜닝한 음성 인식 모델에 전달됩니다. 잘못 분류된 경우엔 질환 구분 없이 전체 데이터를 활용해 튜닝한 모델로 전달됩니다.

</br>

---

### Research Code and Model Repository

#### Research code

- [Preprocess(전처리)](Preprocess)
- [Feature Extraction(특징 추출)](Feature_Extraction)
- [Classfication(질환 분류)](Classfication)
- [Fine-Tuning(모델 튜닝)](Fine-Tuning)
- [Predict(텍스트 추출)](Predict)

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

</br >

### Utterance Testset

#### Predict Utterance Dataset

- [Text_Preprocess](https://drive.google.com/drive/folders/1nj2i7ATyR_r-g64zfu_cPJrdQniVXv36?usp=drive_link)
- [Audio_Preprocess](https://drive.google.com/drive/folders/1gtUJdO5jkNTziiJcdAayRF_uyqLbpMqF?usp=drive_link)

#### Fluency Anakysis Values

음성 데이터를 통해 추출된 유창성 지표 수치와 화자 정보(나이, 성별, 질환)을 포함해 파일로 제공합니다. </br>

- [fluency_analysis](Feature_Extraction/Result/fluency_analysis.csv)

</br >

### Frameworks and Libraries Used

- Local Environment (python 3.10.1, pandas 2.2.3, numpy 1.26.4, openpyxl 3.1.5, noisereduce 3.0.3, praat-parselmouth 0.4.5, librosa 0.11.0)
- Google Colab Environment (python 3.11.13, transformers 4.54.0, torch 2.6.0+cu124, numpy 2.0.2, datasets 3.6.0, evaluate 0.4.4, librosa 0.11.0, nlptutti 0.0.0.10, jiwer 4.0.0)

</br >

### Training and Experimental Setup

- Local (NVIDIA GeForce MX150 & Intel® UHD Graphics 620 GPU, Intel Core i7-8565U @ 1.80GHz (4 cores / 8 threads) CPU, 16GB Ram)
- Google Colab (NVIDIA Tesla T4 GPU, 15GB VRAM, CUDA 12.4, Driver 550.54.15)

---

연구와 관련해서 질문이 있으시다면 아래 메일로 연락주세요.

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>
