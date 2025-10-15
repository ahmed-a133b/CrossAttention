# Literature Review: Cross-Attention Fusion for Plant Disease Classification

This chapter critically examines existing literature relevant to multimodal plant disease classification, with particular focus on attention mechanisms, feature fusion approaches, and deep learning architectures. It identifies key studies, theories, and findings, providing a foundation for the proposed cross-attention research and highlighting its relationship to prior work in plant disease detection.

## 2.1 Related Research

This section reviews 10 significant research items from the systematic review of deep learning techniques for plant diseases (Pacal et al., 2024), focusing on approaches most relevant to our proposed cross-attention fusion methodology.

### 2.1.1 Research Item 1: CNN and ViT-based Grape Disease Classification

**Citation:** Kunduracioglu, I., & Pacal, I. (2024). CNN and ViT-based grape disease classification using PlantVillage and Grapevine datasets, achieving 100% accuracy.

#### 2.1.1.1 Summary of the research item

This study represents one of the most successful applications of Vision Transformer (ViT) architectures in plant disease classification. The researchers implemented both Convolutional Neural Network (CNN) and Vision Transformer models for grape disease detection, utilizing two distinct datasets: PlantVillage and a specialized Grapevine dataset. The approach achieved perfect classification accuracy (100%) across both architectural paradigms. The study demonstrates the effectiveness of transformer-based attention mechanisms in agricultural computer vision tasks, particularly highlighting how self-attention mechanisms in ViTs can capture long-range dependencies in plant leaf images that traditional CNNs might miss. The research validates the potential of attention-based architectures for plant pathology applications.

#### 2.1.1.2 Critical analysis of the research item

**Strengths:** The study's primary strength lies in its comparative analysis of CNN versus ViT architectures, providing valuable insights into the relative performance of traditional convolutional approaches versus modern attention-based methods. The perfect accuracy achievement indicates robust model training and potentially high-quality datasets. The use of multiple datasets (PlantVillage and Grapevine) enhances the generalizability of findings across different data sources.

**Weaknesses:** The 100% accuracy raises concerns about potential overfitting or dataset limitations, as perfect performance is rarely achieved in real-world scenarios. The study appears to focus solely on visual features without exploring multimodal approaches that could incorporate additional contextual information. Limited details are available regarding the specific ViT architecture modifications, training procedures, or cross-validation strategies employed.

#### 2.1.1.3 Relationship to the proposed research work

This research directly relates to our proposed cross-attention approach by demonstrating the effectiveness of attention mechanisms (ViT's self-attention) in plant disease classification. However, it represents a single-modal approach using only visual features. Our proposed cross-attention fusion extends this concept by introducing multimodal learning that combines visual attention (similar to ViT) with textual attention mechanisms, potentially achieving more robust and interpretable disease classification through cross-modal feature interaction.

### 2.1.2 Research Item 2: Attention-Based Mechanisms with ACSPAM-MFFN

**Citation:** Sunil, C.K., Jaidhar, C.D., & Patil, N. (2023). Tomato plant disease classification using multilevel feature fusion with adaptive channel spatial and pixel attention mechanism (ACSPAM-MFFN). PlantVillage dataset, 99.83% accuracy.

#### 2.1.2.1 Summary of the research item

This research introduces the Adaptive Channel Spatial and Pixel Attention Mechanism with Multi-Level Feature Fusion Network (ACSPAM-MFFN) for tomato disease classification. The approach implements sophisticated attention mechanisms that operate at multiple levels: channel-wise attention for feature selection, spatial attention for location-aware processing, and pixel-level attention for fine-grained feature extraction. The multilevel feature fusion strategy combines representations from different network depths to capture both low-level texture details and high-level semantic information. Evaluated on the PlantVillage dataset, the model achieved 99.83% accuracy, demonstrating superior performance compared to traditional CNN approaches without attention mechanisms.

#### 2.1.2.2 Critical analysis of the research item

**Strengths:** The research presents a comprehensive attention framework that addresses multiple aspects of feature extraction and fusion. The multilevel approach is theoretically sound, combining different granularities of attention for holistic feature representation. The high accuracy (99.83%) suggests effective implementation of attention mechanisms. The adaptive nature of the attention components allows for dynamic feature weighting based on input characteristics.

**Weaknesses:** The approach remains within the visual domain and does not explore cross-modal interactions. The complexity of the ACSPAM-MFFN architecture may lead to increased computational overhead and potential overfitting. The evaluation is limited to a single crop type (tomato) and dataset, raising questions about generalizability across different plant species and disease types. The study lacks comparison with other state-of-the-art attention mechanisms.

#### 2.1.2.3 Relationship to the proposed research work

This research is highly relevant as it demonstrates advanced attention mechanism implementation in plant disease classification. The multilevel feature fusion concept aligns with our cross-attention approach, but while ACSPAM-MFFN fuses features within the visual modality, our proposed method extends fusion across different modalities (visual and textual). The attention mechanisms used here provide insights into effective architectural designs that can be adapted for cross-modal attention implementation.

### 2.1.3 Research Item 3: Transfer Learning with DenseNet Architecture

**Citation:** Abbas, A., et al. (2021). DenseNet121 for tomato disease classification using PlantVillage dataset, achieving 99.51% accuracy.

#### 2.1.3.1 Summary of the research item

This study applies DenseNet121, a densely connected convolutional network, to tomato disease classification through transfer learning techniques. DenseNet's architecture features dense connections between layers, where each layer receives feature maps from all preceding layers, promoting feature reuse and gradient flow. The research leverages pre-trained ImageNet weights and fine-tunes the network for plant disease classification on the PlantVillage dataset. The approach achieved 99.51% accuracy, demonstrating the effectiveness of dense connectivity patterns and transfer learning for agricultural computer vision applications.

#### 2.1.3.2 Critical analysis of the research item

**Strengths:** The use of DenseNet121 represents a sophisticated architectural choice that addresses vanishing gradient problems and promotes feature reuse through dense connections. Transfer learning from ImageNet provides a strong foundation for feature extraction, particularly beneficial when agricultural datasets are limited. The high accuracy indicates effective model adaptation from general object recognition to specialized plant pathology tasks.

**Weaknesses:** The approach follows traditional single-modal CNN paradigms without exploring attention mechanisms or multimodal integration. Evaluation is limited to tomato diseases, potentially limiting generalizability across different crop types. The study lacks detailed analysis of which features contribute most to disease classification, missing opportunities for interpretability enhancement that attention mechanisms could provide.

#### 2.1.3.3 Relationship to the proposed research work

While this research represents traditional CNN-based approaches, it provides important baseline performance metrics for comparison with our proposed cross-attention method. The transfer learning strategy used here could be adapted for our multimodal approach, where pre-trained visual encoders (like DenseNet) could be combined with pre-trained language models (like BERT) in our cross-attention framework. The dense connectivity concept also inspires our feature fusion strategies across modalities.

### 2.1.4 Research Item 4: EfficientNet-Based Multi-Plant Classification

**Citation:** Atila, Ü., et al. (2021). EfficientNet for multi-plant disease classification using PlantVillage dataset, achieving 98.42% accuracy.

#### 2.1.4.1 Summary of the research item

This research employs EfficientNet, a compound scaling CNN architecture, for multi-plant disease classification across diverse crop types. EfficientNet's strength lies in its systematic scaling of network depth, width, and resolution using compound coefficients, achieving superior accuracy-efficiency trade-offs. The study addresses the challenging problem of multi-plant classification, where the model must distinguish between diseases across different plant species rather than focusing on a single crop type. The 98.42% accuracy demonstrates EfficientNet's capability to handle diverse visual patterns and disease manifestations across multiple plant types.

#### 2.1.4.2 Critical analysis of the research item

**Strengths:** The multi-plant approach represents more realistic deployment scenarios where agricultural monitoring systems need to handle diverse crop types simultaneously. EfficientNet's compound scaling methodology provides an optimal balance between model performance and computational efficiency, crucial for practical applications. The research addresses scalability concerns inherent in plant disease detection systems.

**Weaknesses:** The study maintains a purely visual approach without incorporating additional modalities that could enhance classification accuracy and robustness. The slightly lower accuracy (98.42%) compared to single-crop studies suggests challenges in multi-plant scenarios that might be addressed through richer feature representations. Limited analysis of cross-plant disease similarities and differences that could inform better architectural designs.

#### 2.1.4.3 Relationship to the proposed research work

This research highlights the complexity of multi-plant disease classification, a challenge our cross-attention approach aims to address through richer multimodal representations. The efficiency considerations of EfficientNet inform our architectural choices for the visual encoder in our cross-attention framework. The multi-plant classification challenge validates the need for more sophisticated feature fusion approaches that our cross-modal attention mechanism provides.

### 2.1.5 Research Item 5: Multi-Plant Detection with Xception Architecture

**Citation:** Saleem, M.H., et al. (2020). Xception model for multi-plant disease classification using PlantVillage dataset, achieving 99.81% accuracy.

#### 2.1.5.1 Summary of the research item

This study implements the Xception architecture, which employs depthwise separable convolutions, for multi-plant disease classification. Xception's key innovation lies in its extreme inception modules that completely separate channel-wise and spatial-wise correlations through depthwise separable convolutions. Applied to the PlantVillage dataset for multi-plant disease detection, the approach achieved exceptional accuracy of 99.81%. The research demonstrates how architectural innovations in convolution operations can significantly impact plant disease classification performance while maintaining computational efficiency.

#### 2.1.5.2 Critical analysis of the research item

**Strengths:** The Xception architecture provides an elegant solution to computational efficiency while maintaining high accuracy through depthwise separable convolutions. The exceptional accuracy (99.81%) across multiple plant types indicates robust feature extraction capabilities. The architectural approach addresses both performance and efficiency concerns relevant to practical agricultural applications.

**Weaknesses:** The study follows conventional single-modal approaches without exploring attention mechanisms or multimodal integration possibilities. Limited discussion of interpretability aspects, which are crucial for agricultural applications where understanding disease characteristics is important. The evaluation focuses primarily on accuracy metrics without considering other important factors like model uncertainty or confidence measures.

#### 2.1.5.3 Relationship to the proposed research work

This research provides important architectural insights for efficient convolution operations that could be incorporated into our visual encoder design within the cross-attention framework. The high performance on multi-plant classification establishes strong baselines for comparison with our proposed multimodal approach. The efficiency considerations of Xception inform our design choices for scalable cross-attention implementations.

### 2.1.6 Research Item 6: ResNet-Based Coffee Disease Classification

**Citation:** Afifi, A., et al. (2020). ResNet18, ResNet34, and ResNet50 for coffee leaf disease classification using PlantVillage and Coffee Leaf datasets, achieving 99% accuracy.

#### 2.1.6.1 Summary of the research item

This comprehensive study evaluates multiple ResNet architectures (ResNet18, ResNet34, and ResNet50) for coffee leaf disease classification across two distinct datasets: PlantVillage and a specialized Coffee Leaf dataset. ResNet's residual connections enable training of very deep networks by addressing vanishing gradient problems through skip connections. The research provides systematic comparison across different network depths, achieving consistent 99% accuracy across architectures and datasets, demonstrating the robustness of residual learning for plant disease classification tasks.

#### 2.1.6.2 Critical analysis of the research item

**Strengths:** The systematic comparison across multiple ResNet variants provides valuable insights into depth-performance relationships in plant disease classification. Evaluation on two different datasets enhances the reliability of findings and demonstrates generalization capabilities. The consistent high performance across architectures indicates robust methodology and dataset quality.

**Weaknesses:** The study remains within traditional CNN paradigms without exploring modern attention mechanisms or multimodal approaches. Limited analysis of what specific features contribute to disease classification success. The focus on coffee diseases, while thorough, limits insights about cross-crop generalization capabilities.

#### 2.1.6.3 Relationship to the proposed research work

ResNet architectures, particularly their residual connections and feature extraction capabilities, serve as excellent candidates for the visual encoder component in our cross-attention framework. The systematic evaluation approach used here informs our experimental design for comparing different architectural choices within our multimodal system. The high baseline performance establishes benchmarks for our cross-attention approach to exceed.

### 2.1.7 Research Item 7: Lightweight MobileNet Approaches

**Citation:** Ayu, M.A., et al. (2021). MobileNetV2 for cassava leaf disease classification using Cassava Leaf Disease Classification dataset, achieving 65.6% accuracy.

#### 2.1.7.1 Summary of the research item

This research applies MobileNetV2, a lightweight CNN architecture designed for mobile and edge deployment, to cassava leaf disease classification. MobileNetV2 employs depthwise separable convolutions and inverted residual blocks to achieve computational efficiency while maintaining reasonable accuracy. The study addresses practical deployment constraints in agricultural settings where computational resources may be limited. However, the achieved accuracy of 65.6% is significantly lower than other approaches, highlighting the trade-offs between model complexity and performance.

#### 2.1.7.2 Critical analysis of the research item

**Strengths:** The focus on lightweight architectures addresses real-world deployment scenarios in agricultural settings with limited computational resources. MobileNetV2's efficiency makes it suitable for mobile and edge applications crucial for field deployment. The research contributes to understanding performance-efficiency trade-offs in agricultural computer vision.

**Weaknesses:** The relatively low accuracy (65.6%) indicates significant limitations in feature extraction capability, potentially due to the model's lightweight nature. The study does not explore techniques to improve accuracy while maintaining efficiency, such as attention mechanisms or knowledge distillation. Limited analysis of failure cases or strategies to enhance lightweight model performance.

#### 2.1.7.3 Relationship to the proposed research work

This research highlights the importance of balancing performance with computational efficiency, a consideration relevant to our cross-attention approach. While our method may be more computationally intensive than MobileNet approaches, the significantly higher accuracy potential justifies the increased complexity. The efficiency considerations inform our design choices for deployment scenarios and potential model compression strategies.

### 2.1.8 Research Item 8: Multi-Level Feature Fusion with DFN-PSAN

**Citation:** Dai, M., et al. (2024). DFN-PSAN model for multi-plant disease detection across PlantVillage, BARI-Sunflower, and FGVC8 datasets, achieving 95.27% accuracy.

#### 2.1.8.1 Summary of the research item

This recent research introduces the Deep Feature Network with Pyramid Self-Attention Network (DFN-PSAN) for interpretable plant disease classification. The approach combines multi-level deep information feature fusion with pyramid attention mechanisms to extract features at different scales and resolutions. The model's interpretability focus addresses critical needs in agricultural applications where understanding decision-making processes is crucial. Evaluation across three diverse datasets (PlantVillage, BARI-Sunflower, and FGVC8) demonstrates cross-dataset generalization capabilities, achieving 95.27% average accuracy.

#### 2.1.8.2 Critical analysis of the research item

**Strengths:** The multi-dataset evaluation approach provides strong evidence for generalization capabilities across different data sources and collection methodologies. The focus on interpretability addresses practical needs in agricultural applications. The pyramid attention mechanism represents sophisticated architectural design for multi-scale feature extraction. The recent publication date indicates incorporation of current state-of-the-art techniques.

**Weaknesses:** While incorporating attention mechanisms, the approach remains single-modal without exploring multimodal integration possibilities. The accuracy (95.27%) is lower than some single-dataset studies, potentially indicating challenges in cross-dataset generalization. Limited details available about the specific attention mechanism implementation and its relationship to interpretability claims.

#### 2.1.8.3 Relationship to the proposed research work

This research is highly relevant as it demonstrates advanced attention mechanism implementation with interpretability focus. The multi-level feature fusion concept directly relates to our cross-attention approach, though DFN-PSAN operates within single modality while our method extends across modalities. The interpretability emphasis validates the importance of explainable AI in agricultural applications, which our cross-attention mechanism can enhance through attention weight visualization across modalities.

### 2.1.9 Research Item 9: Hybrid Deep Learning with DenseNet121

**Citation:** Tiwari, V., et al. (2021). DenseNet121 hybrid approach for multi-plant disease classification using self-collected dataset, achieving 99.97% accuracy.

#### 2.1.9.1 Summary of the research item

This study presents a hybrid deep learning approach based on DenseNet121 for multi-plant disease classification using a self-collected dataset. The hybrid methodology likely combines multiple techniques such as ensemble methods, data augmentation strategies, or multi-stage processing pipelines. The exceptional accuracy of 99.97% represents near-perfect classification performance, suggesting highly effective integration of multiple deep learning techniques. The use of self-collected data provides insights into real-world data collection and annotation challenges in agricultural settings.

#### 2.1.9.2 Critical analysis of the research item

**Strengths:** The near-perfect accuracy demonstrates exceptional technical implementation and potentially innovative hybrid methodologies. The multi-plant approach addresses practical deployment scenarios. Self-collected datasets provide realistic performance evaluation under field conditions rather than standardized benchmark datasets. The hybrid approach suggests sophisticated integration of multiple techniques.

**Weaknesses:** Limited details available about the specific hybrid methodology components and their individual contributions. The exceptional accuracy raises questions about potential overfitting or dataset characteristics. Self-collected datasets may lack the standardization and validation present in established benchmarks, limiting reproducibility and comparison capabilities.

#### 2.1.9.3 Relationship to the proposed research work

This research validates the potential for hybrid approaches in plant disease classification, supporting our multimodal cross-attention strategy as a form of hybrid methodology that combines visual and textual processing. The exceptional performance establishes high benchmarks for our approach to meet or exceed. The hybrid concept aligns with our integration of different modalities and attention mechanisms within a unified framework.

### 2.1.10 Research Item 10: Attention-Embedded ResNet for Disease Detection

**Citation:** Karthik, R., et al. (2020). Attention embedded residual CNN for disease detection in tomato leaves using PlantVillage dataset, achieving 98% accuracy.

#### 2.1.10.1 Summary of the research item

This research integrates attention mechanisms into ResNet architecture for tomato leaf disease detection, representing an early adoption of attention-based approaches in plant pathology. The attention-embedded residual CNN combines the gradient flow benefits of ResNet's skip connections with attention mechanisms that focus on relevant image regions for disease identification. Applied to tomato disease classification using the PlantVillage dataset, the approach achieved 98% accuracy, demonstrating the effectiveness of attention integration within established CNN architectures.

#### 2.1.10.2 Critical analysis of the research item

**Strengths:** The research represents pioneering work in combining attention mechanisms with established CNN architectures for plant disease detection. The integration approach maintains the proven benefits of ResNet while adding attention capabilities for improved feature selection. The methodology provides a foundation for more sophisticated attention-based approaches in agricultural computer vision.

**Weaknesses:** The attention mechanism appears to be single-modal, focusing only on spatial attention within visual features without exploring cross-modal possibilities. The accuracy (98%) is lower than some recent approaches, suggesting room for improvement in attention mechanism design. Limited analysis of attention visualization or interpretability aspects.

#### 2.1.10.3 Relationship to the proposed research work

This research directly validates the effectiveness of attention mechanisms in plant disease classification, providing foundational support for our cross-attention approach. While this study implements spatial attention within visual modality, our proposed method extends attention across different modalities (visual and textual). The ResNet backbone used here could serve as the visual encoder component in our cross-attention framework, building upon proven architectural foundations.

## 2.2 Analysis Summary of Research Items

The following table summarizes the critical analysis of the research items discussed, highlighting their key contributions, limitations, and relationships to the proposed cross-attention fusion approach:

| Research Item | Architecture | Accuracy | Key Strengths | Main Limitations | Relevance to Cross-Attention |
|---------------|--------------|----------|---------------|------------------|------------------------------|
| 1. Kunduracioglu & Pacal (2024) | CNN + ViT | 100% | Comparative analysis, Perfect accuracy, Multi-dataset | Potential overfitting, Single-modal | Validates attention effectiveness (ViT self-attention) |
| 2. Sunil et al. (2023) | ACSPAM-MFFN | 99.83% | Multi-level attention, Adaptive mechanisms | Single-modal, High complexity | Advanced attention framework design |
| 3. Abbas et al. (2021) | DenseNet121 | 99.51% | Dense connections, Transfer learning | Traditional CNN, Limited interpretability | Potential visual encoder backbone |
| 4. Atila et al. (2021) | EfficientNet | 98.42% | Multi-plant capability, Efficiency balance | Single-modal, Lower accuracy | Multi-plant classification challenges |
| 5. Saleem et al. (2020) | Xception | 99.81% | Efficient convolutions, Multi-plant | Single-modal, Limited interpretability | Efficient architectural patterns |
| 6. Afifi et al. (2020) | ResNet variants | 99% | Systematic evaluation, Multi-dataset | Traditional CNN, Single-modal | Proven visual encoder foundation |
| 7. Ayu et al. (2021) | MobileNetV2 | 65.6% | Lightweight design, Mobile deployment | Low accuracy, Limited capability | Efficiency considerations |
| 8. Dai et al. (2024) | DFN-PSAN | 95.27% | Cross-dataset generalization, Interpretability | Single-modal, Lower accuracy | Multi-level fusion concepts |
| 9. Tiwari et al. (2021) | Hybrid DenseNet | 99.97% | Near-perfect accuracy, Hybrid approach | Limited methodology details | Hybrid methodology validation |
| 10. Karthik et al. (2020) | Attention-ResNet | 98% | Early attention adoption, CNN integration | Single-modal spatial attention | Foundational attention implementation |

### Key Insights from Literature Analysis:

1. **Attention Mechanism Potential**: Multiple studies (Items 1, 2, 8, 10) demonstrate the effectiveness of attention mechanisms in plant disease classification, with ViT self-attention and spatial attention showing significant improvements over traditional CNNs.

2. **Single-Modal Limitation**: All reviewed studies operate within single modalities (visual-only), representing a significant research gap that our cross-attention approach addresses through multimodal integration.

3. **High Performance Benchmarks**: Several studies achieve >99% accuracy, establishing high performance standards that our cross-attention method must meet or exceed to demonstrate value.

4. **Multi-Plant Classification Challenges**: Studies focusing on multiple plant types (Items 4, 5, 8, 9) show varying performance levels, indicating the complexity of generalized plant disease detection that multimodal approaches could address.

5. **Interpretability Requirements**: Recent works (Items 8, 2) emphasize interpretability needs in agricultural applications, which our cross-attention mechanism can enhance through attention weight visualization across modalities.

6. **Architecture Foundation**: Proven architectures like ResNet, DenseNet, and EfficientNet provide strong foundations for visual encoders in multimodal systems, as demonstrated by consistently high performance across studies.

This literature analysis reveals a clear research gap in multimodal approaches for plant disease classification, validating the novelty and potential impact of our proposed cross-attention fusion methodology that combines visual and textual modalities through sophisticated attention mechanisms.