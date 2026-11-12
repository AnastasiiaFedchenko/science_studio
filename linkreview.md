# Обзор литературы

## Введение

Распознавание эмоций по выражению лица (Facial Emotion Recognition, FER) остаётся активно развивающимся направлением компьютерного зрения, востребованным в человеко-машинном взаимодействии, социальной робототехнике и клинической диагностике. В работах [1–7] представлены систематические обзоры последних лет, которые подчёркивают, что область сместилась в сторону глубокого обучения, однако сохраняется фундаментальный компромисс между точностью и вычислительной эффективностью, особенно для развёртывания на edge-устройствах. Данный обзор систематизирует существующие подходы, выявляет их ограничения и формулирует исследовательский пробел, на который направлена настоящая работа.

## Классификация подходов

### Подход 1: Глубокие сверточные нейронные сети (CNN)

Доминирующий класс методов основан на сквозной обработке пиксельного представления изображения свёрточными сетями. На бенчмарке FER-2013 [8] современные гибридные архитектуры достигают высокой точности: в работе [9] предложена архитектура CNN-DET (комбинация CNN с Deep Extra-Trees), которая сообщает о **85.32%** на стандартном тестовом разбиении. Другие работы [10, 12] сообщают о 82.6–84.2% для ResNet-50 и EfficientNet-B0, а также о 72.93% для кастомных CNN с аугментацией данных. В работе [13] предложен обобщаемый фреймворк на базе глубокого обучения для мониторинга когнитивных нарушений.

**Ограничение:** вычислительная стоимость. Несмотря на достигнутые успехи, такие модели остаются вычислительно тяжёлыми и плохо интерпретируемыми [1, 6]. Даже «компактные» решения для edge-устройств требуют специализированных архитектур: например, лёгкая гибридная модель LightFER [11], позиционируемая как эффективная, всё же опирается на комбинацию CNN и Transformer. Легковесные фреймворки на геометрических признаках (например, для ассистивной робототехники [28]) показывают, что можно обходиться без тяжёлых CNN, но требуют тщательной нормализации landmarks. Таким образом, CNN обеспечивают качество, но не решают задачу предельно лёгкой и интерпретируемой системы.

### Подход 2: Геометрические признаки на основе ключевых точек и blendshapes

Альтернативное направление использует координаты ключевых точек лица или вычисляемые на их основе признаки. Библиотека **MediaPipe Face Landmarker** [14] извлекает 478 трёхмерных landmarks и, что принципиально, **52 blendshape-коэффициента** — непрерывные оценки активации мимических мышц, стандартизированные по топологии ARKit.

**Сильная сторона:** низкая вычислительная сложность и устойчивость к вариациям освещения и поворота головы. В работе [15] подтверждается, что 52 коэффициента MediaPipe могут быть экспертно-валидированно сопоставлены с Action Units, что делает их интерпретируемым и масштабируемым признаковым пространством. В исследовании [16] демонстрируется, что blendshape-последовательности от MediaPipe, обработанные LSTM, достигают **71% accuracy** и **62% F1** на FER-2013 при существенно меньших вычислительных затратах, чем тяжёлые CNN. Другие работы используют blendshapes для предсказания статических оценок [17] или для рендеринга эмоций в реальном времени [18].

**Ограничение:** большинство работ либо «заворачивает» blendshapes в нейросетевые архитектуры (LSTM, Transformer), либо использует их как промежуточное представление, но не проверяет целенаправленно, что даёт **минимальный классический пайплайн** (blendshapes + традиционный алгоритм) и можно ли его улучшить простыми модификациями.

### Подход 3: Сравнительные исследования классических алгоритмов

Наименее исследованная ниша — систематическое сравнение традиционных классификаторов на blendshape- или landmark-признаках. В работе [20] на FER-2013 с MediaPipe Face Mesh показано, что **Random Forest** достигает ~53% accuracy, **CNN** — ~54%, и что классические модели остаются конкурентоспособными в лёгком режиме. В исследовании [21] на FER-2013 с 478 MediaPipe-landmarks проведено сравнение MNB, Decision Trees, KNN, Random Forest, SVM, XGBoost и Logistic Regression: лучший результат показала **логистическая регрессия (65.57%)**, за ней XGBoost (61.24%). Обзор [22] и работа [23] также подтверждают, что SVM и Random Forest могут давать сопоставимые результаты на геометрических признаках при правильном подборе гиперпараметров. На HuggingFace доступна модель `Nonovogo/face-emotions2` [19], использующая 52 blendshape-оценки MediaPipe с Random Forest на FER-2013 и достигающая точности **51.36%** — это воспроизводимый лёгкий baseline, однако его потенциал систематически не исследован в литературе.

## Сравнительный анализ и критический синтез

Сопоставление трёх направлений выявляет следующую картину. Тяжёлые CNN [9–13] обеспечивают максимальную точность (82–85% и выше на FER-2013), но плохо масштабируются на устройства без GPU и не дают интерпретируемых промежуточных представлений. Blendshapes с нейросетевыми классификаторами (LSTM [16], Transformer [18]) занимают промежуточное положение: они легче CNN и достигают приемлемого качества (71%), но всё ещё требуют обучения глубокой модели. Наконец, blendshapes с классическими классификаторами [19–21] — это минимальное по сложности решение, но его потенциал **систематически не исследован**: сравнительные работы либо используют полные landmark-наборы (478 точек), а не компактные 52 blendshape-признака, либо ограничиваются воспроизведением одного классификатора без попыток целенаправленного улучшения.

Отдельно стоит отметить проблему временной динамики. Хотя blendshape-временные ряды содержат информацию о динамике мимики, большинство статических бенчмарков (включая FER-2013) не позволяют её использовать. Работы, исследующие межкадровые разности, относятся к смежным задачам: в [24] Differential Temporal Transformer использует forward/backward-разности (Δf, Δb) для подавления фонового смещения и захвата динамических паттернов в задаче динамического FER; в [26] патентное решение применяет motion stream через межкадровые разности признаков для обнаружения микровыражений; в [25, 27] методы используют временные производные для микровыражений и оценки уровня депрессии. Однако вопрос о том, даёт ли добавление простых межкадровых разностей измеримое улучшение в **лёгком blendshape-пайплайне для статической классификации эмоций**, остаётся открытым.

Легковесные edge-ориентированные FER-системы [28–30] подтверждают актуальность поиска компромисса: даже «эффективные» решения на геометрических признаках требуют сложной инженерии признаков и нормализации (например, в [29] используются объяснимые ML-модели на 3D-landmarks, а в [30] — система на YOLO11 с MediaPipe FaceMesh, работающая на 24 FPS), однако они фокусируются на нормализации и отборе landmarks, а не на выборе классификатора. Кроме того, нейросетевые компактные модели остаются тяжелее классического пайплайна.

## Вывод: исследовательский пробел

Существующие работы либо максимизируют точность за счёт вычислительной сложности (CNN), либо используют blendshapes внутри нейросетевых архитектур (LSTM, Transformer), либо ограничиваются воспроизведением базового классического пайплайна без попыток его улучшения. **Ни одна из проанализированных работ не проверяет целенаправленно гипотезу о том, что простая замена классификатора (Random Forest → SVM / Logistic Regression) на фиксированных 52 blendshape-признаках MediaPipe и/или добавление межкадровых разностных признаков способны обеспечить измеримый прирост Macro-F1 на FER-2013 без выхода за рамки лёгкого и интерпретируемого пайплайна.**

Этот пробел и определяет цель настоящей работы: воспроизвести базовую линию (Random Forest + blendshapes, ~51.36%) и экспериментально проверить две модификации — замену классификатора и добавление временных разностных признаков — с оценкой их влияния на метрики качества в рамках строго контролируемого вычислительного эксперимента.

---

## Список литературы (30 позиций, все использованы в тексте)

1. A Systematic Literature Review on Deep Learning Based Multimodal Facial Expression Recognition: Trends, Gaps, and Future Opportunities // IEEE. — 2025. (Конференция 25–26 сентября 2025, Джакарта)
2. From Low-Resolution to Surveillance: A Systematic Literature Review on Facial Expression Recognition Techniques // IEEE. — 2026.
3. Leveraging State-of-the-art Deep Learning Advancements for Emotion Detection: A Comprehensive Review and Insights // Google Scholar. — 2026 (обзор FER 2022–2025).
4. Deep learning-based facial expression recognition for the elderly: A systematic review // ScienceDirect. — 2026.
5. Artificial intelligence for emotion recognition: Techniques, trends, and challenges // ScienceDirect. — 2026.
6. From Handcrafted to Structural Hybridization: A Comprehensive Survey of Hybrid Deep Learning Architectures for Facial Emotion Recognition // IEEE. — 2026.
7. Facial Emotion Recognition (FER) in multimodal settings: A systematic literature review // Garuda. — 2026.
8. Carrier P.L., Courville A., Goodfellow I.J., Mirza M., Bengio Y. FER-2013 Face Database // Tech. Rep. 1365, Université de Montréal. — 2013.
9. CNN-DET: A hybrid deep learning architecture for emotion recognition // ScienceDirect (Elsevier). — 2026. — FER-2013: 85.32% на стандартном тесте.
10. Facial emotion recognition and summarization in mock interviews using CNN-transformer hybrids // ScienceDirect. — 2026. — ResNet-50 (82.6%), EfficientNet-B0 (84.2%) на FER-2013.
11. LightFER: A Lightweight Hybrid CNN-Transformer based Model for Efficient Facial Emotion Recognition // IEEE. — 2026.
12. A custom CNN architecture with comprehensive data augmentation // Zenodo. — 2025. — 72.93% на FER-2013.
13. Enhanced Facial Expression Recognition Using Deep Learning: A Generalizable Framework with Potential for Cognitive Impairment Monitoring // ScienceDirect. — 2026.
14. Google AI Edge. Face Landmarker — MediaPipe Documentation. — 2026. (52 blendshape scores)
15. Blendshape features meet action units: a clinical mapping for enhancing facial expression analysis // ScienceDirect (Elsevier). — 2026. — DOI: 10.1016/j.chb.2026.1995
16. Attrah S. Emotion estimation from video footage with LSTM // Frontiers in Neurorobotics. — 2026. — Vol. 19. — Art. 1678984. DOI: 10.3389/fnbot.2025.1678984 (BlendFER-Lite: 71% accuracy, 62% F1 на FER-2013).
17. Emotion-modulated interaction regression predicts static facial blendshape scores // Scientific Reports (Nature). — 2026. — 52-мерные MediaPipe/ARKit-style blendshape-векторы.
18. Real-Time Neural Emotion Rendering for Embodied Conversational Agents Using LSTM Dynamics and ONNX BlendShape Inference // IEEE. — 2026.
19. Nonovogo/face-emotions2 — MediaPipe Face Emotion Classifier (Random Forest + 52 Blendshapes, FER-2013, accuracy 51.36%) // HuggingFace. — 2026.
20. Ozhan C. Emotion Recognition with Deep Learning-Based Facial Expressions: Comparative Analysis of Algorithms. — Masters thesis, Dublin, National College of Ireland, 2025. — CNN ~54%, Random Forest ~53% на FER-2013 + MediaPipe Face Mesh.
21. Automatic Emotion Detection from Facial Features Extraction and Machine Learning Models // IEEE Xplore. — 2025. — FER-2013 + MediaPipe (478 landmarks): LR 65.57%, XGBoost 61.24%.
22. Facial Emotion Recognition – A Comprehensive Review of Deep Learning and Traditional Learning Approaches with Emerging Challenges // TCI-Thaijo. — 2026.
23. SVM, Random Forest, KNN, Decision Tree с Grid Search // Zenodo. — 2025.
24. Dynamic Facial Expression Recognition of Learners via Adaptive Global Attention and Differential Temporal Transformer // IET Research (Wiley). — 2026.
25. DBDE-Net: Dual-branch detail-enhanced network for micro-expression recognition // ScienceDirect. — 2026.
26. CN122116445A — Face expression recognition method (patent). — 2026. — Motion stream через inter-frame feature differences + adaptive gating.
27. A multi-scale temporal difference network for depression level assessment from facial video // CNKI. — 2026.
28. A Lightweight Vision-Based Emotion Sensing Framework for Assistive Healthcare Robotics // Sensors (Basel). — 2026. — Vol. 26(9), 2865. DOI: 10.3390/s26092865.
29. Efficient Emotion Recognition Using Media-Pipe 3D Landmarks and Explainable Machine Learning Models // IEEE. — 2025.
30. A Real time Face Emotion Detection System Based on YOLO11 / MediaPipe FaceMesh // Zenodo. — 2026. — Edge-развёртывание, 24 FPS.