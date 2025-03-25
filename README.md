# Natural Language Processing for Human Resources: A Survey

This repository provides a list of papers and resources related to the application of Natural Language Processing (NLP) in Human Resources (HR). The list will be updated regularly.

### Taxonomy Generation

#### [ESCO: Boosting Job Matching in Europe with Semantic Interoperability](https://ieeexplore.ieee.org/document/6928765) (le Vrang et al., 2014)

- **Authors:** le Vrang, Martin; Papantoniou, Agis; Pauwels, Erika; Fannes, Pieter; Vandensteen, Dominique; De Smedt, Johan
- **Year:** 2014
- **Publication:** Computer
- **Summary:** This article introduces ESCO, a multilingual classification system that aims to remove communication obstacles in the European labor market. ESCO connects information on occupations and qualifications with skills and competences based on a three-pillar framework (occupations, skills and competences, and qualifications).

#### [LinkedIn Skills: Large-Scale Topic Extraction and Inference](https://doi.org/10.1145/2645710.2645729) (Bastian et al., 2014)

- **Authors:** Bastian, Mathieu; Hayes, Matthew; Vaughan, William; Shah, Sam; Skomoroch, Peter; Kim, Hyungjin; Uryasev, Sal; Lloyd, Christopher
- **Year:** 2014
- **Publication:** Proceedings of the 8th ACM Conference on Recommender systems
- **Summary:** This paper presents the development of a large-scale IE pipeline for LinkedIn's "Skills and Expertise" feature. The pipeline includes constructing a folksonomy of skills and expertise and implementing an inference and recommender system for skills. The paper also discusses applications like Endorsements, which allows members to tag themselves with topics representing their areas of expertise and for their connections to provide social proof of that member's competence in that topic.

#### [SKILL: A System for Skill Identification and Normalization](https://ojs.aaai.org/index.php/AAAI/article/view/19064) (Zhao et al., 2015)

- **Authors:** Zhao, Meng; Javed, Faizan; Jacob, Ferosh; McNair, Matt
- **Year:** 2015
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper presents a system for skill identification and normalization, which is crucial for the predictive analysis of labor market dynamics. The system has two components: 1) Skills taxonomy generation, which defines and develops a taxonomy of professional skills; 2) Skills tagging, which recognizes and normalizes relevant skills in input text. The system has been applied in various big data and business intelligence applications for workforce analytics and career track projections at CareerBuilder.

#### [Generating Unified Candidate Skill Graph for Career Path Recommendation](https://ieeexplore.ieee.org/document/8637575) (Gugnani et al., 2018)

- **Authors:** Gugnani, Akshay; Reddy Kasireddy, Vinay Kumar; Ponnalagu, Karthikeyan
- **Year:** 2018
- **Publication:** 2018 IEEE International Conference on Data Mining Workshops
- **Summary:** This paper proposes a system to recommend career paths using skill graphs instead of job titles, making recommendations more flexible across different organizations and industries. It extracts skills from a user's profile using an Open IE pipeline and maps them into a unified skill graph, capturing both time-based and contextual relationships. An evaluation using industry-scale data showed promising results, with precision at 80.54% and recall at 86.44%.

#### [Beyond human-in-the-loop: scaling occupation taxonomy at Indeed](https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_2.pdf) (Tu et al., 2022)

- **Authors:** Tu, Suyi; Cannon, Olivia
- **Year:** 2022
- **Publication:** Proceedings of the 2nd Workshop on Recommender Systems for Human Resources
- **Summary:** This paper describes how Indeed scaled its occupation taxonomy using a combination of machine learning models and subject matter experts. The improved system allowed for faster and more efficient job matching across international markets without compromising on quality. It also explores the role of experts beyond simple labeling, proposing an expert-in-the-loop approach for scaling taxonomy.

### Job Classification

#### [Carotene: A Job Title Classification System for the Online Recruitment Domain](https://ieeexplore.ieee.org/document/7184892) (Javed et al., 2015)

- **Authors:** Javed, Faizan; Luo, Qinlong; McNair, Matt; Jacob, Ferosh; Zhao, Meng; Kang, Tae Seung
- **Year:** 2015
- **Publication:** 2015 IEEE First International Conference on Big Data Computing Service and Applications
- **Summary:** This paper presents Carotene, a machine learning-based semi-supervised job title classification system. Carotene leverages a varied collection of classification and clustering tools and techniques to tackle the challenges of designing a scalable classification system for a large taxonomy of job categories.

#### [Skills and Vacancy Analysis with Data Mining Techniques](https://www.mdpi.com/2227-9709/2/4/31) (Wowczko, 2015)

- **Authors:** Wowczko, Izabela A.
- **Year:** 2015
- **Publication:** Informatics
- **Summary:** This paper presents a method for classifying jobs and skills in job advertisements using exsiting tools.

#### [Flexible Job Classification with Zero-Shot Learning](https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_8.pdf) (Lake, 2022)

- **Authors:** Lake, Thom
- **Year:** 2022
- **Publication:** Proceedings of the 2nd Workshop on Recommender Systems for Human Resources
- **Summary:** This paper explores zero-shot learning for flexible multi-label document classification. Results show that zero-shot classifiers outperform traditional models when budgets for labeled data are limited, suggesting a better use of resources by focusing on incomplete class sets. Additionally, integrating filter/re-rank techniques greatly reduces computational costs with minimal performance loss, showcasing zero-shot learning's potential for adaptable taxonomy management.

### Resume Classification

#### [Competence-Level Prediction and Resume & Job Description Matching Using Context-Aware Transformer Models](https://aclanthology.org/2020.emnlp-main.679) (Li et al., 2020)

- **Authors:** Li, Changmao; Fisher, Elaine; Thomas, Rebecca; Pittard, Steve; Hertzberg, Vicki; Choi, Jinho D.
- **Year:** 2020
- **Publication:** Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)
- **Summary:** This paper explores using transformer models to improve resume screening efficiency by predicting candidate suitability for Clinical Research Coordinator (CRC) positions. It introduces two tasks: classifying resumes into CRC experience levels and matching resumes with job descriptions. The proposed models achieve promising accuracy, suggesting their usefulness in real-world HR applications, especially given the challenges even experts face in distinguishing between closely related experience levels.

#### [DGL4C: a Deep Semi-supervised Graph Representation Learning Model for Resume Classification](https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_5.pdf) (Inoubli et al., 2022)

- **Authors:** Inoubli, Wissem; Brun, Armelle
- **Year:** 2022
- **Publication:** Proceedings of the 2nd Workshop on Recommender Systems for Human Resources
- **Summary:** This paper proposes DGL4C, a semi-supervised graph deep learning model designed for classifying resumes to match job offers. By representing resumes and job offers as graphs, DGL4C captures important relationships and learns relevant representations for classification. Experimental results demonstrate that DGL4C outperforms traditional deep learning models like sBERT, improving precision and accuracy in human resource applications.

#### [Identifying and Improving Disability Bias in GPT-Based Resume Screening](https://doi.org/10.1145/3630106.3658933) (Glazko et al., 2024)

- **Authors:** Glazko, Kate; Mohammed, Yusuf; Kosa, Ben; Potluri, Venkatesh; Mankoff, Jennifer
- **Year:** 2024
- **Publication:** Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper analyzes the presence of disability bias in GPT-4’s resume evaluations. By comparing standard resumes with versions containing disability-related achievements, the study reveals noticeable biases against applicants with these additions. The research demonstrates that custom training focused on diversity, equity, and inclusion (DEI) principles can significantly reduce this bias, pointing to potential solutions for fairer AI-based hiring practices.

### Resume Assessment

#### [DOMFN: A Divergence-Orientated Multi-Modal Fusion Network for Resume Assessment](https://dl.acm.org/doi/10.1145/3503161.3548203) (Yang et al., 2022)

- **Authors:** Yang, Yang; Zhang, Jingshuai; Gao, Fan; Gao, Xiaoru; Zhu, Hengshu
- **Year:** 2022
- **Publication:** Proceedings of the 30th ACM International Conference on Multimedia
- **Summary:** This paper proposes a method to resume assessment, addressing the issue of cross-modal divergence in multi-modal fusion. The Divergence-Orientated Multi-Modal Fusion Network (DOMFN) adapts to when fusion should occur by evaluating the divergence between different modalities, ensuring the most effective prediction method is chosen. Experiments on real-world datasets show that DOMFN outperforms existing models and offers a better understanding of when and why multi-modal fusion works in specific contexts.

### Biography Classification

#### [Effective Controllable Bias Mitigation for Classification and Retrieval using Gate Adapters](https://aclanthology.org/2024.eacl-long.150) (Masoudian et al., 2024)

- **Authors:** Masoudian, Shahed; Volaucnik, Cornelia; Schedl, Markus; Rekabsaz, Navid
- **Year:** 2024
- **Publication:** Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics
- **Summary:** This paper presents ConGater, a new modular mechanism that allows users to control the level of bias reduction in LMs during inference. By adjusting sensitivity parameters, ConGater provides a smooth transition from a biased to a debiased model, making it easier to balance performance and fairness. Through experiments on classification and retrieval tasks, ConGater outperforms other methods by maintaining higher task performance while reducing bias, offering flexibility and interpretability for practical applications.

### Skill Extraction And Classification

#### [SKILL: A System for Skill Identification and Normalization](https://ojs.aaai.org/index.php/AAAI/article/view/19064) (Zhao et al., 2015)

- **Authors:** Zhao, Meng; Javed, Faizan; Jacob, Ferosh; McNair, Matt
- **Year:** 2015
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper presents a system for skill identification and normalization, which is crucial for the predictive analysis of labor market dynamics. The system has two components: 1) Skills taxonomy generation, which defines and develops a taxonomy of professional skills; 2) Skills tagging, which recognizes and normalizes relevant skills in input text. The system has been applied in various big data and business intelligence applications for workforce analytics and career track projections at CareerBuilder.

#### [Learning Representations for Soft Skill Matching](https://doi.org/10.1007/978-3-030-11027-7_15) (Sayfullina et al., 2018)

- **Authors:** Sayfullina, Luiza; Malmi, Eric; Kannala, Juho
- **Year:** 2018
- **Publication:** Analysis of Images, Social Networks and Texts
- **Summary:** This work proposes a phrase-matching approach to detect and disambiguate soft skill mentions in job advertisements, addressing the challenge of false positives when soft skills describe entities other than candidates. By framing the task as a binary text classification problem, the authors explore input representations such as soft skill masking and tagging to enhance contextual understanding. Among various neural network models tested, the tagging-based LSTM approach achieved the best balance of precision and recall, highlighting its effectiveness for accurate soft skill identification.

#### [Responsible team players wanted: an analysis of soft skill requirements in job advertisements](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-019-0190-z) (Calanca et al., 2019)

- **Authors:** Calanca, Federica; Sayfullina, Luiza; Minkus, Lara; Wagner, Claudia; Malmi, Eric
- **Year:** 2019
- **Publication:** EPJ Data Science
- **Summary:** This paper examines the growing emphasis on soft skills in job advertisements and their implications for labor market outcomes, particularly concerning gender and wage inequality. Using a combination of crowdsourcing, text mining, and interdisciplinary insights, it highlights that soft skills are especially prominent in low-wage, female-dominated professions and can predict gender composition within job categories. The findings underscore that “female” soft skills often attract wage penalties, contributing to occupational gender segregation and wage disparities.

#### [Retrieving Skills from Job Descriptions: A Language Model Based Extreme Multi-label Classification Framework](https://aclanthology.org/2020.coling-main.513) (Bhola et al., 2020)

- **Authors:** Bhola, Akshay; Halder, Kishaloy; Prasad, Animesh; Kan, Min-Yen
- **Year:** 2020
- **Publication:** Proceedings of the 28th International Conference on Computational Linguistics
- **Summary:** This paper presents a deep learning framework leveraging BERT for extreme multi-label classification (XMLC) to identify job skills from job descriptions. The model addresses the challenge of incomplete skill enumeration in job postings by using a Correlation Aware Bootstrapping process to account for semantic relationships and co-occurrences among skills, achieving significant improvements in recall and nDCG over baseline methods. The dataset and implementation are made publicly available to support future research and replication.

#### [Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning](https://aclanthology.org/2022.lrec-1.46) (Zhang et al., 2022)

- **Authors:** Zhang, Mike; Jensen, Kristian Nørgaard; Plank, Barbara
- **Year:** 2022
- **Publication:** Proceedings of the Thirteenth Language Resources and Evaluation Conference
- **Summary:** This paper presents a new dataset of Danish job postings, Kompetencer, annotated for skill classification. It uses distant supervision with the ESCO taxonomy API to provide more detailed labels and explores both zero-shot and few-shot learning methods. The study shows that RemBERT performs better than other models in both settings, demonstrating its effectiveness for fine-grained skill classification.

#### [JobXMLC: EXtreme Multi-Label Classification of Job Skills with Graph Neural Networks](https://aclanthology.org/2023.findings-eacl.163) (Goyal et al., 2023)

- **Authors:** Goyal, Nidhi; Kalra, Jushaan; Sharma, Charu; Mutharaju, Raghava; Sachdeva, Niharika; Kumaraguru, Ponnurangam
- **Year:** 2023
- **Publication:** Findings of the Association for Computational Linguistics: EACL 2023
- **Summary:** This paper presents JobXMLC, a framework that improves the prediction of missing job skills in job descriptions using a graph-based approach. By incorporating job-job and job-skill relationships, the model uses graph neural networks and skill attention to make accurate skill predictions. The proposed framework significantly outperforms existing methods in terms of both precision and recall, while also being much faster and scalable.

#### [Large Language Models as Batteries-Included Zero-Shot ESCO Skills Matchers](https://ceur-ws.org/Vol-3490/RecSysHR2023-paper_8.pdf) (Clavié et al., 2023)

- **Authors:** Clavié, Benjamin; Soulié, Guillaume
- **Year:** 2023
- **Publication:** Proceedings of the 3rd Workshop on Recommender Systems for Human Resources
- **Summary:** This workproposes a method to extract skills from job descriptions and match them with ESCO with LLMs. By generating synthetic training data and utilizing both a similarity retriever and re-ranking with GPT-4, the system achieves significantly improved results compared to previous methods.

#### [Extreme Multi-Label Skill Extraction Training using Large Language Models](https://ai4hrpes.github.io/ecmlpkdd2023/papers/ai4hrpes2023_paper_173.pdf) (Decorte et al., 2023)

- **Authors:** Decorte, Jens-Joris; Verlinden, Severine; Hautte, Jeroen Van; Deleu, Johannes; Develder, Chris; Demeester, Thomas
- **Year:** 2023
- **Publication:** Proceedings of the 4th International workshop on AI for Human Resources and Public Employment Services
- **Summary:** This work focuses on using LLMs to address the challenge of skill extraction from online job ads, a key task in labor market analysis. The authors propose a cost-effective method to generate synthetic labeled data for training, overcoming the lack of large labeled datasets. Through contrastive learning, their approach shows a significant improvement in performance, outperforming previous methods based on literal skill matching.

#### [ESCOXLM-R: Multilingual Taxonomy-driven Pre-training for the Job Market Domain](https://aclanthology.org/2023.acl-long.662) (Zhang et al., 2023)

- **Authors:** Zhang, Mike; van der Goot, Rob; Plank, Barbara
- **Year:** 2023
- **Publication:** Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
- **Summary:** This paper presents ESCOXLM-R, a multilingual language model tailored for job market-related tasks. It is pre-trained using the ESCO taxonomy and performs well on a variety of job-specific tasks, such as skill extraction and job title classification, across multiple languages. The model shows superior performance compared to existing models, especially for short text spans and entity-level tasks.

#### [Entity Linking in the Job Market Domain](https://aclanthology.org/2024.findings-eacl.28) (Zhang et al., 2024)

- **Authors:** Zhang, Mike; Goot, Rob; Plank, Barbara
- **Year:** 2024
- **Publication:** Findings of the Association for Computational Linguistics: EACL 2024
- **Summary:** This work addresses the challenge of linking mentions of occupational skills to the ESCO taxonomy. The authors adapt two neural models, a bi-encoder and an autoregressive model, and test them using both synthetic and human-annotated data. Results indicate that both models can effectively connect implicit skill mentions to the correct taxonomy entries, with the bi-encoder performing better under strict parameters and the autoregressive model excelling in more lenient evaluations.

#### [NNOSE: Nearest Neighbor Occupational Skill Extraction](https://aclanthology.org/2024.eacl-long.35) (Zhang et al., 2024)

- **Authors:** Zhang, Mike; Goot, Rob; Kan, Min-Yen; Plank, Barbara
- **Year:** 2024
- **Publication:** Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics
- **Summary:** This paper addresses the challenge of extracting occupational skills from diverse job description datasets. It introduces the Nearest Neighbor Occupational Skill Extraction (NNOSE) method, which uses an external datastore to unify multiple datasets and enhance the retrieval of similar skills without the need for extra fine-tuning. The method shows significant improvements, especially in predicting less common skills, achieving up to a 30% increase in span-F1 scores in scenarios involving multiple datasets.

#### [Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings](https://aclanthology.org/2024.nlp4hr-1.1) (Senger et al., 2024)

- **Authors:** Senger, Elena; Zhang, Mike; Goot, Rob; Plank, Barbara
- **Year:** 2024
- **Publication:** Proceedings of the First Workshop on Natural Language Processing for Human Resources
- **Summary:** This survey paper reviews recent advancements in skill extraction and classification from job postings. It covers key techniques, datasets, and terminology used in this field, filling the gap left by previous studies. By organizing and clarifying these concepts, the paper aims to provide a useful resource for researchers and practitioners working in this area.

#### [JobSkape: A Framework for Generating Synthetic Job Postings to Enhance Skill Matching](https://aclanthology.org/2024.nlp4hr-1.4) (Magron et al., 2024)

- **Authors:** Magron, Antoine; Dai, Anna; Zhang, Mike; Montariol, Syrielle; Bosselut, Antoine
- **Year:** 2024
- **Publication:** Proceedings of the First Workshop on Natural Language Processing for Human Resources
- **Summary:** This paper proposes JobSkape, a framework for generating synthetic job postings to improve skill matching. It introduces SkillSkape, an open-source dataset for skill-matching tasks, addressing previous dataset limitations such as having only one skill per sentence and being too short. The authors also provide a multi-step pipeline for skill extraction and matching using LLMs, showing improved results compared to traditional methods.

### Job Title Normalization

#### [Learning Job Title Representation from Job Description Aggregation Network](https://aclanthology.org/2024.findings-acl.77) (Laosaengpha et al., 2024)

- **Authors:** Laosaengpha, Napat; Tativannarat, Thanit; Piansaddhayanon, Chawan; Rutherford, Attapol; Chuangsuwanich, Ekapol
- **Year:** 2024
- **Publication:** Findings of the Association for Computational Linguistics ACL 2024
- **Summary:** This paper proposes a new approach to understanding job titles by analyzing entire job descriptions rather than just extracted skills. It introduces a Job Description Aggregator to process detailed descriptions and uses a contrastive learning method to capture the connection between titles and their descriptions. The method outperformed traditional skill-based models in various evaluation settings.

#### [Job2Vec: Job Title Benchmarking with Collective Multi-View Representation Learning](https://dl.acm.org/doi/10.1145/3357384.3357825) (Zhang et al., 2019)

- **Authors:** Zhang, Denghui; Liu, Junming; Zhu, Hengshu; Liu, Yanchi; Wang, Lichen; Wang, Pengyang; Xiong, Hui
- **Year:** 2019
- **Publication:** Proceedings of the 28th ACM International Conference on Information and Knowledge Management
- **Summary:** This paper presents Job2Vec, a novel approach to Job Title Benchmarking (JTB) that leverages a data-driven solution using a large-scale Job Title Benchmarking Graph (Job-Graph) constructed from extensive career records. The method addresses challenges like non-standard job titles, missing information, and limited job transition data by employing collective multi-view representation learning, examining graph topology, semantic meaning, transition balance, and transition duration. Extensive experiments demonstrate that the proposed method effectively predicts links in the Job-Graph, enabling accurate matching of similar-level job titles across companies.

#### [Improving Word Embeddings through Iterative Refinement of Word- and Character-level Models](https://aclanthology.org/2020.coling-main.104) (Ha et al., 2020)

- **Authors:** Ha, Phong; Zhang, Shanshan; Djuric, Nemanja; Vucetic, Slobodan
- **Year:** 2020
- **Publication:** Proceedings of the 28th International Conference on Computational Linguistics
- **Summary:** This paper introduces an iterative refinement algorithm that enhances word and character-level embeddings to better handle rare and OOV words. The approach involves training a character-level neural network to replicate standard word embeddings, then iteratively improving both models. Results show the proposed method outperforms baselines and proves effective in job title normalization within the e-recruitment domain.

#### [JobBERT: Understanding Job Titles through Skills](https://feast-ecmlpkdd.github.io/archive/2021/papers/FEAST2021_paper_6.pdf) (Decorte et al., 2021)

- **Authors:** Decorte, Jens-Joris; Hautte, Jeroen Van; Demeester, Thomas; Develder, Chris
- **Year:** 2021
- **Publication:** FEAST: International Workshop on Fair, Effective, and Sustainable Talent Management using Data Science
- **Summary:** This paper introduces JobBERT, a model that uses a pre-trained language model combined with skill co-occurrence data to better represent job titles. By incorporating skill labels from job postings, JobBERT improves accuracy in normalizing job titles compared to general sentence models. The authors also present a new evaluation benchmark for this task.

#### [Occupational profiling driven by online job advertisements: Taking the data analysis and processing engineering technicians as an example](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253308) (Cao et al., 2021)

- **Authors:** Cao, Lina; Zhang, Jian; Ge, Xinquan; Chen, Jindong
- **Year:** 2021
- **Publication:** PLOS ONE
- **Summary:** This paper demonstrates a method to enhance occupational profiling by using online job advertisements, focusing specifically on data analysis and processing engineering technicians (DAPET). The approach involves a multi-step process: first, identifying relevant job ads using text similarity algorithms; second, employing TextCNN for precise classification; and third, extracting named entities related to specialties and skills. The resulting occupation characteristics create a dynamic and detailed vocational profile that addresses the limitations of traditional profiling systems.

#### [Towards Job-Transition-Tag Graph for a Better Job Title Representation Learning](https://aclanthology.org/2022.findings-naacl.164) (Zhu et al., 2022)

- **Authors:** Zhu, Jun; Hudelot, Celine
- **Year:** 2022
- **Publication:** Findings of the Association for Computational Linguistics: NAACL 2022
- **Summary:** This paper proposes a new method to improve job title representation learning by creating a Job-Transition-Tag Graph. Unlike traditional Job-Transition Graphs, this graph includes both job titles and relevant tags, reducing sparsity and improving the quality of representations. Experiments on two datasets (job title classification and next-job prediction) demonstrate the effectiveness of this approach.

#### [JAMES: Normalizing Job Titles with Multi-Aspect Graph Embeddings and Reasoning](https://ieeexplore.ieee.org/abstract/document/10302559) (Yamashita et al., 2023)

- **Authors:** Yamashita, Michiharu; Shen, Jia Tracy; Tran, Thanh; Ekhtiari, Hamoon; Lee, Dongwon
- **Year:** 2023
- **Publication:** 2023 IEEE 10th International Conference on Data Science and Advanced Analytics
- **Summary:** This paper proposes JAMES, a methodfor Job Title Normalization. JAMES uses three distinct embeddings—graph, contextual, and syntactic—to capture different aspects of a job title and employs a multi-aspect co-attention mechanism to combine them effectively. The model outperforms existing methods in job title normalization, achieving notable improvements in precision and relevance in real-world datasets.

#### [ESCOXLM-R: Multilingual Taxonomy-driven Pre-training for the Job Market Domain](https://aclanthology.org/2023.acl-long.662) (Zhang et al., 2023)

- **Authors:** Zhang, Mike; van der Goot, Rob; Plank, Barbara
- **Year:** 2023
- **Publication:** Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
- **Summary:** This paper presents ESCOXLM-R, a multilingual language model tailored for job market-related tasks. It is pre-trained using the ESCO taxonomy and performs well on a variety of job-specific tasks, such as skill extraction and job title classification, across multiple languages. The model shows superior performance compared to existing models, especially for short text spans and entity-level tasks.

### Information Extraction

#### [Relational Learning of Pattern-Match Rules for Information Extraction](https://aclanthology.org/W97-1002/) (Califf et al., 1997)

- **Authors:** Califf, Mary Elaine; Mooney, Raymond J.
- **Year:** 1997
- **Publication:** CoNLL97: Computational Natural Language Learning
- **Summary:** This paper introduces a machine learning-based system that automatically learns rules to extract specific information from documents, demonstrating promising results in processing job postings.

#### [Multilingual Generation and Summarization of Job Adverts: the TREE Project](https://aclanthology.org/A97-1040) (Somers et al., 1997)

- **Authors:** Somers, Harold; Black, Bill; Nivre, Joakim; Lager, Torbjorn; Multari, Annarosa; Gilardoni, Luca; Ellman, Jeremy; Rogers, Alex
- **Year:** 1997
- **Publication:** Fifth Conference on Applied Natural Language Processing
- **Summary:** This paper introduces a multilingual job search system, which parses job ads into structured data and generate summaries of the job ads for an input user query. This work introduces example-based techniques to handle lexical diversity of job ads.

#### [Information Extraction via Double Classification](nan) (De Sitter et al., 2003)

- **Authors:** De Sitter, An; Daelemans, Walter
- **Year:** 2003
- **Publication:** Proceedings of the International Workshop on Adaptive Text Extraction and Mining
- **Summary:** This paper presents a novel approach to IE by employing a two-step classification process to identify and extract relevant information from documents. The first classification loop focuses on identifying pertinent sentences in a document, while the second loop performs word-level analysis for more granular extraction. The approach is tested on the Software Jobs corpus, with an extensive evaluation discussing various parameters and highlighting the significant impact of the evaluation method on the results.

#### [Multi-level Boundary Classification for Information Extraction](https://doi.org/10.1007/978-3-540-30115-8_13) (Finn et al., 2004)

- **Authors:** Finn, Aidan; Kushmerick, Nicholas
- **Year:** 2004
- **Publication:** Proceedings of the 15th European Conference on Machine Learning
- **Summary:** This work applies SVMs with various feature sets to IE tasks, showing its competitive performance compared with existing, specialized IE algorithms. A novel method is proposed that uses a two-level ensemble of classifiers to enhance recall without sacrificing precision, achieving improved results on benchmark tasks.

#### [Resume Information Extraction with Cascaded Hybrid Model](https://aclanthology.org/P05-1062) (Yu et al., 2005)

- **Authors:** Yu, Kun; Guan, Gang; Zhou, Ming
- **Year:** 2005
- **Publication:** Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics
- **Summary:** This paper introduces a cascaded information extraction framework for IE from resumes. A two-pass model segments and labels resume sections before identifying specific details, achieving superior accuracy compared to non-hierarchical models.

#### [LinkedIn Skills: Large-Scale Topic Extraction and Inference](https://doi.org/10.1145/2645710.2645729) (Bastian et al., 2014)

- **Authors:** Bastian, Mathieu; Hayes, Matthew; Vaughan, William; Shah, Sam; Skomoroch, Peter; Kim, Hyungjin; Uryasev, Sal; Lloyd, Christopher
- **Year:** 2014
- **Publication:** Proceedings of the 8th ACM Conference on Recommender systems
- **Summary:** This paper presents the development of a large-scale IE pipeline for LinkedIn's "Skills and Expertise" feature. The pipeline includes constructing a folksonomy of skills and expertise and implementing an inference and recommender system for skills. The paper also discusses applications like Endorsements, which allows members to tag themselves with topics representing their areas of expertise and for their connections to provide social proof of that member's competence in that topic.

#### [Skills and Vacancy Analysis with Data Mining Techniques](https://www.mdpi.com/2227-9709/2/4/31) (Wowczko, 2015)

- **Authors:** Wowczko, Izabela A.
- **Year:** 2015
- **Publication:** Informatics
- **Summary:** This paper presents a method for classifying jobs and skills in job advertisements using exsiting tools.

#### [Generating Unified Candidate Skill Graph for Career Path Recommendation](https://ieeexplore.ieee.org/document/8637575) (Gugnani et al., 2018)

- **Authors:** Gugnani, Akshay; Reddy Kasireddy, Vinay Kumar; Ponnalagu, Karthikeyan
- **Year:** 2018
- **Publication:** 2018 IEEE International Conference on Data Mining Workshops
- **Summary:** This paper proposes a system to recommend career paths using skill graphs instead of job titles, making recommendations more flexible across different organizations and industries. It extracts skills from a user's profile using an Open IE pipeline and maps them into a unified skill graph, capturing both time-based and contextual relationships. An evaluation using industry-scale data showed promising results, with precision at 80.54% and recall at 86.44%.

#### [Salience and Market-aware Skill Extraction for Job Targeting](https://doi.org/10.1145/3394486.3403338) (Shi et al., 2020)

- **Authors:** Shi, Baoxu; Yang, Jaewon; Guo, Feng; He, Qi
- **Year:** 2020
- **Publication:** Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
- **Summary:** This paper introduces Job2Skills, a system designed to extract job-relevant skills by considering both their prominence in job postings and market trends such as supply and demand. Unlike traditional methods, Job2Skills evaluates the significance of skills dynamically, leading to better job recommendations and more accurate skill suggestions. Deployed on LinkedIn, it has improved job application rates and reduced skill suggestion rejections, benefiting millions of job postings globally.

#### [Analyzing the relationship between information technology jobs advertised on-line and skills requirements using association rules](https://beei.org/index.php/EEI/article/view/2590) (Patacsil et al., 2021)

- **Authors:** Patacsil, Frederick F.; Acosta, Michael
- **Year:** 2021
- **Publication:** Bulletin of Electrical Engineering and Informatics
- **Summary:** This paper proposes a method for analyzing the relationship between IT jobs and their required skills by utilizing job postings and applying association rule mining techniques. By using the FP-growth algorithm to discover frequent patterns and relationships in job postings, the study identifies how specific skills are linked to IT job requirements. The findings highlight gaps between educational training and industry demands, suggesting potential areas for curriculum development and policy changes by the Philippine government to better align with the labor market's needs.

#### [A Survey on Skill Identification From Online Job Ads](https://ieeexplore.ieee.org/document/9517309) (Khaouja et al., 2021)

- **Authors:** Khaouja, Imane; Kassou, Ismail; Ghogho, Mounir
- **Year:** 2021
- **Publication:** IEEE Access
- **Summary:** This survey paper reviews over 108 studies on skill identification in online job advertisements. It evaluates various approaches, including skill bases, extraction methods, granularity, and sector-specific insights, highlighting current applications and trends. The paper concludes with insights into future research directions, offering a thorough comparison of methodologies and contributions in this field.

#### [Occupational profiling driven by online job advertisements: Taking the data analysis and processing engineering technicians as an example](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253308) (Cao et al., 2021)

- **Authors:** Cao, Lina; Zhang, Jian; Ge, Xinquan; Chen, Jindong
- **Year:** 2021
- **Publication:** PLOS ONE
- **Summary:** This paper demonstrates a method to enhance occupational profiling by using online job advertisements, focusing specifically on data analysis and processing engineering technicians (DAPET). The approach involves a multi-step process: first, identifying relevant job ads using text similarity algorithms; second, employing TextCNN for precise classification; and third, extracting named entities related to specialties and skills. The resulting occupation characteristics create a dynamic and detailed vocational profile that addresses the limitations of traditional profiling systems.

#### [Resume Parsing Framework for E-recruitment](https://doi.org/10.1109/IMCOM53663.2022.9721762) (Sajid et al., 2022)

- **Authors:** Sajid, Hira; Kanwal, Javeria; Bhatti, Saeed Ur Rehman; Qureshi, Saad Ali; Basharat, Amna; Hussain, Shujaat; Khan, Kifayat Ullah
- **Year:** 2022
- **Publication:** 2022 16th International Conference on Ubiquitous Information Management and Communication (IMCOM)
- **Summary:** This paper presents a framework for improving the accuracy of resume parsing. The proposed method extracts and classifies blocks of information in resumes and uses named entity recognition along with ontology enrichment to gather key data. This approach aims to address the challenges of varied resume formats and incomplete knowledge in previous systems, offering a more reliable solution for identifying the best candidates.

#### [Design of Negative Sampling Strategies for Distantly Supervised Skill Extraction](https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf) (Decorte et al., 2022)

- **Authors:** Decorte, Jens-Joris; Hautte, Jeroen Van; Deleu, Johannes; Develder, Chris; Demeester, Thomas
- **Year:** 2022
- **Publication:** Proceedings of the 2nd Workshop on Recommender Systems for Human Resources
- **Summary:** This study explores automated skill extraction in the job market using distant supervision. It proposes various negative sampling strategies to improve the detection of implicitly mentioned skills and demonstrates that selecting related skills from the ESCO taxonomy enhances performance. The study also introduces a new evaluation benchmark and dataset to advance research in this area.

#### [“FIJO”: a French Insurance Soft Skill Detection Dataset](https://caiac.pubpub.org/pub/72bhunl6/release/1) (Beauchemin et al., 2022)

- **Authors:** Beauchemin, David; Laumonier, Julien; Ster, Yvan Le; Yassine, Marouane
- **Year:** 2022
- **Publication:** Proceedings of the 35th Canadian Conference on Artificial Intelligence
- **Summary:** This paper introduces FIJO, a new public dataset focused on French insurance job offers with annotated soft skills. It explores the dataset’s features and evaluates skill detection algorithms based on BiLSTM and BERT, showing strong performance from the BERT model. The study also discusses errors made by the best model.

#### [Development of a Benchmark Corpus to Support Entity Recognition in Job Descriptions](https://aclanthology.org/2022.lrec-1.128) (Green et al., 2022)

- **Authors:** Green, Thomas; Maynard, Diana; Lin, Chenghua
- **Year:** 2022
- **Publication:** Proceedings of the Thirteenth Language Resources and Evaluation Conference
- **Summary:** This paper introduces a benchmark corpus for entity recognition in job descriptions, including an annotated dataset, schema, and baseline model. The dataset, containing over 18k entities in five categories, aims to improve the extraction of skills and qualifications from job postings. The benchmark CRF model demonstrates a starting point with an F1 score of 0.59, offering a foundation for future research and practical applications like job recommendation systems.

#### [Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning](https://aclanthology.org/2022.lrec-1.46) (Zhang et al., 2022)

- **Authors:** Zhang, Mike; Jensen, Kristian Nørgaard; Plank, Barbara
- **Year:** 2022
- **Publication:** Proceedings of the Thirteenth Language Resources and Evaluation Conference
- **Summary:** This paper presents a new dataset of Danish job postings, Kompetencer, annotated for skill classification. It uses distant supervision with the ESCO taxonomy API to provide more detailed labels and explores both zero-shot and few-shot learning methods. The study shows that RemBERT performs better than other models in both settings, demonstrating its effectiveness for fine-grained skill classification.

#### [ResuFormer: Semantic Structure Understanding for Resumes via Multi-Modal Pre-training](https://ieeexplore.ieee.org/document/10184685) (Yao et al., 2023)

- **Authors:** Yao, Kaichun; Zhang, Jingshuai; Qin, Chuan; Song, Xin; Wang, Peng; Zhu, Hengshu; Xiong, Hui
- **Year:** 2023
- **Publication:** 2023 IEEE 39th International Conference on Data Engineering
- **Summary:** This paper proposes ResuFormer, a model aimed at better understanding the structure of resumes by incorporating both textual and multi-modal data, such as layout and visual information. The model focuses on two key tasks: classifying resume blocks and extracting information within those blocks. Experiments on real-world datasets show that ResuFormer outperforms existing methods and provides robust performance even with limited labeled data.

#### [Large Language Models as Batteries-Included Zero-Shot ESCO Skills Matchers](https://ceur-ws.org/Vol-3490/RecSysHR2023-paper_8.pdf) (Clavié et al., 2023)

- **Authors:** Clavié, Benjamin; Soulié, Guillaume
- **Year:** 2023
- **Publication:** Proceedings of the 3rd Workshop on Recommender Systems for Human Resources
- **Summary:** This workproposes a method to extract skills from job descriptions and match them with ESCO with LLMs. By generating synthetic training data and utilizing both a similarity retriever and re-ranking with GPT-4, the system achieves significantly improved results compared to previous methods.

#### [Résumé Parsing as Hierarchical Sequence Labeling: An Empirical Study](http://arxiv.org/abs/2309.07015) (Retyk et al., 2023)

- **Authors:** Retyk, Federico; Fabregat, Hermenegildo; Aizpuru, Juan; Taglio, Mariana; Zbib, Rabih
- **Year:** 2023
- **Publication:** Proceedings of the 3rd Workshop on Recommender Systems for Human Resources
- **Summary:** This paper presents a approach for resume parsing by treating it as a hierarchical sequence labeling task at both the line and token levels. The study uses a diverse set of high-quality resume corpora across multiple languages and demonstrates that the proposed models perform better than existing methods. Additionally, an analysis of model performance and resource efficiency is provided, with insights into how the models can be deployed in real-world applications.

### Document Segmentation

#### [Resume Information Extraction with Cascaded Hybrid Model](https://aclanthology.org/P05-1062) (Yu et al., 2005)

- **Authors:** Yu, Kun; Guan, Gang; Zhou, Ming
- **Year:** 2005
- **Publication:** Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics
- **Summary:** This paper introduces a cascaded information extraction framework for IE from resumes. A two-pass model segments and labels resume sections before identifying specific details, achieving superior accuracy compared to non-hierarchical models.

### Job Search

#### [Semantic Matching Between Job Offers and Job Search Requests](https://aclanthology.org/C90-1014) (Vega, 1990)

- **Authors:** Vega, Jose
- **Year:** 1990
- **Publication:** COLING 1990 Volume 1: Papers presented to the 13th International Conference on Computational Linguistics
- **Summary:** This paper describes a knowledge-based system to retrieve job advertisements that match an input user query or CV.

#### [Multilingual Generation and Summarization of Job Adverts: the TREE Project](https://aclanthology.org/A97-1040) (Somers et al., 1997)

- **Authors:** Somers, Harold; Black, Bill; Nivre, Joakim; Lager, Torbjorn; Multari, Annarosa; Gilardoni, Luca; Ellman, Jeremy; Rogers, Alex
- **Year:** 1997
- **Publication:** Fifth Conference on Applied Natural Language Processing
- **Summary:** This paper introduces a multilingual job search system, which parses job ads into structured data and generate summaries of the job ads for an input user query. This work introduces example-based techniques to handle lexical diversity of job ads.

#### [A survey of job recommender systems](https://academicjournals.org/journal/IJPS/article-abstract/B19DCA416592) (Al-Otaibi et al., 2012)

- **Authors:** Al-Otaibi, Shaha T.; Ykhlef, Mourad
- **Year:** 2012
- **Publication:** International Journal of Physical Sciences
- **Summary:** This survey paper reviews the challenges of traditional e-recruiting platforms and how they miss potential candidates due to outdated search techniques. It explores how recommender systems, commonly used in e-commerce, can address these challenges by offering personalized job matching. Various approaches for designing such systems in the e-recruiting context are discussed.

#### [MEET: A Generalized Framework for Reciprocal Recommender Systems](https://doi.org/10.1145/2396761.2396770) (Li et al., 2012)

- **Authors:** Li, Lei; Li, Tao
- **Year:** 2012
- **Publication:** Proceedings of the 21st ACM international conference on Information and knowledge management
- **Summary:** This paper introduces a framework for reciprocal recommendation that models user correlations through a bipartite graph to balance individual preferences and overall network quality. Here, text data is converted into a vector using TFIDF to represent the feature of a job seeker or a job posting.

#### [A recommender system for job seeking and recruiting website](https://dl.acm.org/doi/10.1145/2487788.2488092) (Lu et al., 2013)

- **Authors:** Lu, Yao; El Helou, Sandy; Gillet, Denis
- **Year:** 2013
- **Publication:** Proceedings of the 22nd International Conference on World Wide Web
- **Summary:** This paper presents a hybrid recommender system for job seeking and recruiting websites. The system exploits the job and user profiles and the actions undertaken by users in order to generate personalized recommendations of candidates and jobs. These pieces of information is represented as a graph with edges weighted by content similarity. A preliminary evaluation is conducted based on simulated data and production data from a job hunting website in Switzerland.

#### [LinkedIn Skills: Large-Scale Topic Extraction and Inference](https://doi.org/10.1145/2645710.2645729) (Bastian et al., 2014)

- **Authors:** Bastian, Mathieu; Hayes, Matthew; Vaughan, William; Shah, Sam; Skomoroch, Peter; Kim, Hyungjin; Uryasev, Sal; Lloyd, Christopher
- **Year:** 2014
- **Publication:** Proceedings of the 8th ACM Conference on Recommender systems
- **Summary:** This paper presents the development of a large-scale IE pipeline for LinkedIn's "Skills and Expertise" feature. The pipeline includes constructing a folksonomy of skills and expertise and implementing an inference and recommender system for skills. The paper also discusses applications like Endorsements, which allows members to tag themselves with topics representing their areas of expertise and for their connections to provide social proof of that member's competence in that topic.

#### [Search by Ideal Candidates: Next Generation of Talent Search at LinkedIn](nan) (Ha-Thuc et al., 2016)

- **Authors:** Ha-Thuc, Viet; Xu, Ye; Kanduri, Satya Pradeep; Wu, Xianren; Dialani, Vijay; Yan, Yan; Gupta, Abhishek; Sinha, Shakti
- **Year:** 2016
- **Publication:** Proceedings of the 25th International Conference Companion on World Wide Web
- **Summary:** This paper describes LinkedIn's new talent search system, "Search by Ideal Candidates," which simplifies translating complex hiring criteria into effective search queries. Users input examples of suitable candidates, and the system automatically generates and refines queries, displaying transparent results. This interactive approach allows recruiters to adjust searches easily, ensuring better alignment between desired qualifications and candidate rankings.

#### [FA*IR: A Fair Top-k Ranking Algorithm](https://dl.acm.org/doi/10.1145/3132847.3132938) (Zehlike et al., 2017)

- **Authors:** Zehlike, Meike; Bonchi, Francesco; Castillo, Carlos; Hajian, Sara; Megahed, Mohamed; Baeza-Yates, Ricardo
- **Year:** 2017
- **Publication:** Proceedings of the 2017 ACM on Conference on Information and Knowledge Management
- **Summary:** This work proposes a solution to the Fair Top-k Ranking problem, where the goal is to select a subset of k candidates from a larger pool, optimizing utility while ensuring group fairness. An efficient algorithm is introduced to produce the Fair Top-k Ranking, which is tested on multiple datasets, showing small distortions compared to rankings that prioritize utility alone, and is the first to address bias mitigation in ranked lists through statistical tests.

#### [Personalized Job Recommendation System at LinkedIn: Practical Challenges and Lessons Learned](https://doi.org/10.1145/3109859.3109921) (Kenthapadi et al., 2017)

- **Authors:** Kenthapadi, Krishnaram; Le, Benjamin; Venkataraman, Ganesh
- **Year:** 2017
- **Publication:** Proceedings of the Eleventh ACM Conference on Recommender Systems
- **Summary:** The paper discusses LinkedIn's personalized job recommendation system and highlights the challenges such as the efficiency of algorithms, the heterogeneity of data, and the reciprocal nature of job matching. The study also outlines lessons learned from real-world implementation, focusing on the trade-offs between personalization and scalability in the job recommendation process.

#### [Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning](https://dl.acm.org/doi/10.1145/3234465) (Zhu et al., 2018)

- **Authors:** Zhu, Chen; Zhu, Hengshu; Xiong, Hui; Ma, Chao; Xie, Fang; Ding, Pengliang; Li, Pan
- **Year:** 2018
- **Publication:** ACM Transactions on Management Information Systems
- **Summary:** This paper proposes a novel CNN-based model, the Person-Job Fit Neural Network (PJFNN), to quantitatively measure and match personal competencies to job requirements. PJFNN operates as a bipartite neural network that learns joint representations from historical job applications, allowing it to evaluate not only the overall suitability of a candidate for a job but also identify specific job requirements that the candidate fulfills. Extensive experiments using a large-scale real-world dataset demonstrate PJFNN's effectiveness in predicting Person-Job Fit, and the study offers data visualizations for gaining insights into job requirements and candidate competencies.

#### [Document-based Recommender System for Job Postings using Dense Representations](https://aclanthology.org/N18-3027) (Elsafty et al., 2018)

- **Authors:** Elsafty, Ahmed; Riedl, Martin; Biemann, Chris
- **Year:** 2018
- **Publication:** Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 3 (Industry Papers)
- **Summary:** This paper presents a document-based recommender system for job postings, leveraging dense vector representations to enhance similarity detection between job advertisements. By combining job titles and full-text descriptions, the system prioritizes important words and achieves better similarity rankings. The proposed method improves user engagement, as evidenced by an 8.0% increase in the click-through rate during an online A/B test with over 1 million users.

#### [Towards Deep and Representation Learning for Talent Search at LinkedIn](https://doi.org/10.1145/3269206.3272030) (Ramanath et al., 2018)

- **Authors:** Ramanath, Rohan; Inan, Hakan; Polatkan, Gungor; Hu, Bo; Guo, Qi; Ozcaglar, Cagri; Wu, Xianren; Kenthapadi, Krishnaram; Geyik, Sahin Cem
- **Year:** 2018
- **Publication:** Proceedings of the 27th ACM International Conference on Information and Knowledge Management
- **Summary:** This paper explores the application of deep and representation learning models to improve LinkedIn's talent search and recommendation systems, addressing the limitations of traditional linear and ensemble tree models in capturing complex feature interactions. Key contributions include learning semantic representations of sparse entities (e.g., recruiter, candidate, and skill IDs) using neural networks and employing deep models to predict recruiter engagement and candidate response. It demonstrates the effectiveness of learning-to-rank approaches for deep models and discusses the challenges of transitioning to fully deep architectures in multi-faceted search engines.

#### [Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach](https://dl.acm.org/doi/10.1145/3209978.3210025) (Qin et al., 2018)

- **Authors:** Qin, Chuan; Zhu, Hengshu; Xu, Tong; Zhu, Chen; Jiang, Liang; Chen, Enhong; Xiong, Hui
- **Year:** 2018
- **Publication:** The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval
- **Summary:** This paper proposes the Ability-aware Person-Job Fit Neural Network (APJFNN) for Person-Job Fit. The propomed method employs LSTM to learn semantic representations of job requirements and candidates' experiences, incorporating hierarchical attention strategies to weigh the relevance of abilities and experiences effectively. Extensive experiments on a real-world dataset demonstrate the APJFNN model's superior performance and interpretability compared to traditional baselines.

#### [Job Recommender Systems: A Survey](https://ieeexplore.ieee.org/document/8960231) (Dhameliya et al., 2019)

- **Authors:** Dhameliya, Juhi; Desai, Nikita
- **Year:** 2019
- **Publication:** 2019 Innovations in Power and Advanced Computing Technologies (i-PACT)
- **Summary:** This survey paper provides an overview of the challenges and advancements in (pre-Transformers) job recommender systems used in online recruitment platforms.

#### [Explaining and exploring job recommendations: a user-driven approach for interacting with knowledge-based job recommender systems](https://dl.acm.org/doi/abs/10.1145/3298689.3347001) (Gutiérrez et al., 2019)

- **Authors:** Gutiérrez, Francisco; Charleer, Sven; De Croon, Robin; Htun, Nyi Nyi; Goetschalckx, Gerd; Verbert, Katrien
- **Year:** 2019
- **Publication:** Proceedings of the 13th ACM Conference on Recommender Systems
- **Summary:** This paper introduces the Labor Market Explorer, an interactive dashboard that helps job seekers navigate the labor market based on their skills. Developed with input from job seekers and mediators, the tool shows personalized job recommendations and the skills needed for each role. Evaluations suggest it effectively supports users in understanding job opportunities and finding relevant positions, regardless of their background or age.

#### [Predicting Online Performance of Job Recommender Systems With Offline Evaluation](https://dl.acm.org/doi/abs/10.1145/3298689.3347032) (Mogenet et al., 2019)

- **Authors:** Mogenet, Adrien; Pham, Tuan Anh Nguyen; Kazama, Masahiro; Kong, Jialin
- **Year:** 2019
- **Publication:** Proceedings of the 13th ACM Conference on Recommender Systems
- **Summary:** This work compares offline and online evaluation methods for job recommender systems at Indeed. It discusses how rare feedback signals complicate accurate performance assessment and describes the metrics used to measure success. The authors explore how offline metrics align with online results, aiming to improve decision-making efficiency in deploying new models.

#### [PrivateJobMatch: a privacy-oriented deferred multi-match recommender system for stable employment](https://dl.acm.org/doi/abs/10.1145/3298689.3346983) (Saini et al., 2019)

- **Authors:** Saini, Amar; Rusu, Florin; Johnston, Andrew
- **Year:** 2019
- **Publication:** Proceedings of the 13th ACM Conference on Recommender Systems
- **Summary:** This paper presents PrivateJobMatch, a system that aims to improve job matching while protecting user privacy. By adapting the Gale-Shapley algorithm and using machine learning, it creates stable job matches without requiring extensive personal data. Tests with real and simulated data show that the system outperforms traditional job markets using only partial user preference rankings.

#### [Should we Embed? A Study on the Online Performance of Utilizing Embeddings for Real-Time Job Recommendations](https://dl.acm.org/doi/abs/10.1145/3298689.3346989) (Lacic et al., 2019)

- **Authors:** Lacic, Emanuel; Reiter-Haas, Markus; Duricic, Tomislav; Slawicek, Valentin; Lex, Elisabeth
- **Year:** 2019
- **Publication:** Proceedings of the 13th ACM Conference on Recommender Systems
- **Summary:** This paper explores the use of embeddings for job recommendations on an Austrian job platform. For recommending similar jobs, the best results are achieved using embeddings based on a user's latest interaction. For personalizing homepage job listings, combining interaction frequency and recency produces the most effective recommendations.

#### [Tripartite Vector Representations for Better Job Recommendation](https://ceur-ws.org/Vol-2512/paper2.pdf) (Liu et al., 2019)

- **Authors:** Liu, Mengshu; Wang, Jingya; Abdelfatah, Kareem; Korayem, Mohammed
- **Year:** 2019
- **Publication:** Proceedings of the 1st International Workshop on Challenges and Experiences from Data Integration to Knowledge Graphs
- **Summary:** This work improves job recommenders by utilizing tripartite vector representations derived from three information graphs: job-job, skill-skill, and job-skill. The joint representation of job titles, skills, and location is created in a shared latent space, enabling better matching of candidates with job postings. Experimental results show that the proposed method outperforms baseline approaches in terms of recommendation relevance.

#### [Domain Adaptation for Person-Job Fit with Transferable Deep Global Match Network](https://aclanthology.org/D19-1487) (Bian et al., 2019)

- **Authors:** Bian, Shuqing; Zhao, Wayne Xin; Song, Yang; Zhang, Tao; Wen, Ji-Rong
- **Year:** 2019
- **Publication:** Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)
- **Summary:** This paper addresses the task of person-job fit by domain adaptation. The proposed method uses a deep global match network to capture semantic interactions between job postings and resumes and implements domain (=industry) adaptation at three levels: sentence-level representation, sentence-level match, and global match. Experiments on a large, multi-domain dataset demonstrate the model's effectiveness, particularly in low-data scenarios.

#### [Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search](https://doi.org/10.1145/3292500.3330691) (Geyik et al., 2019)

- **Authors:** Geyik, Sahin Cem; Ambler, Stuart; Kenthapadi, Krishnaram
- **Year:** 2019
- **Publication:** Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
- **Summary:** This paper proposes a framework to quantify and reduce algorithmic bias in ranking systems used for search and recommendations, with a focus on fairness concerning attributes like gender and age. It introduces algorithms for fairness-aware re-ranking, aiming to achieve fairness goals such as equality of opportunity and demographic parity. The framework was applied to LinkedIn Talent Search, demonstrating a significant improvement in fairness metrics without compromising business outcomes, leading to its global deployment.

#### [ResumeGAN: An Optimized Deep Representation Learning Framework for Talent-Job Fit via Adversarial Learning](https://dl.acm.org/doi/10.1145/3357384.3357899) (Luo et al., 2019)

- **Authors:** Luo, Yong; Zhang, Huaizheng; Wen, Yonggang; Zhang, Xinwen
- **Year:** 2019
- **Publication:** Proceedings of the 28th ACM International Conference on Information and Knowledge Management
- **Summary:** This work proposes ResumeGAN, a deep learning framework designed to improve talent-job matching by integrating various types of information and utilizing adversarial learning to generate more expressive representations.

#### [Towards Effective and Interpretable Person-Job Fitting](https://dl.acm.org/doi/10.1145/3357384.3357949) (Le et al., 2019)

- **Authors:** Le, Ran; Hu, Wenpeng; Song, Yang; Zhang, Tao; Zhao, Dongyan; Yan, Rui
- **Year:** 2019
- **Publication:** Proceedings of the 28th ACM International Conference on Information and Knowledge Management
- **Summary:** This paper presents an Interpretable Person-Job Fit (IPJF) model that incorporates the perspectives of both employers and job seekers by using multi-task optimization and deep interactive representation learning. Experiments show that the IPJF model outperforms existing methods, providing both more accurate job recommendations and clear, interpretable reasons for those recommendations.

#### [PostAc : A Visual Interactive Search, Exploration, and Analysis Platform for PhD Intensive Job Postings](https://aclanthology.org/P19-3008) (Xu et al., 2019)

- **Authors:** Xu, Chenchen; Mewburn, Inger; Grant, Will J; Suominen, Hanna
- **Year:** 2019
- **Publication:** Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations
- **Summary:** This work presents PostAc, an online platform designed to make the job market for Australian PhD graduates more visible, particularly for non-academic roles. The platform uses a ranking model based on job description text to highlight in-demand PhD skills and provides detailed job market insights, including location, sector, and wage information. The research aims to bridge the gap between PhD graduates and employers, helping job seekers better navigate opportunities outside academia.

#### [An Enhanced Neural Network Approach to Person-Job Fit in Talent Recruitment](https://dl.acm.org/doi/10.1145/3376927) (Qin et al., 2020)

- **Authors:** Qin, Chuan; Zhu, Hengshu; Xu, Tong; Zhu, Chen; Ma, Chao; Chen, Enhong; Xiong, Hui
- **Year:** 2020
- **Publication:** ACM Transactions on Information Systems
- **Summary:** proposes a new neural network framework, TAPJFNN, for improving person-job matching. The model uses topic-based attention mechanisms to analyze job requirements and candidate experiences, drawing on historical application data. Experiments show that TAPJFNN outperforms existing methods and offers better interpretability in applications like talent sourcing and job recommendations.

#### [Competence-Level Prediction and Resume & Job Description Matching Using Context-Aware Transformer Models](https://aclanthology.org/2020.emnlp-main.679) (Li et al., 2020)

- **Authors:** Li, Changmao; Fisher, Elaine; Thomas, Rebecca; Pittard, Steve; Hertzberg, Vicki; Choi, Jinho D.
- **Year:** 2020
- **Publication:** Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)
- **Summary:** This paper explores using transformer models to improve resume screening efficiency by predicting candidate suitability for Clinical Research Coordinator (CRC) positions. It introduces two tasks: classifying resumes into CRC experience levels and matching resumes with job descriptions. The proposed models achieve promising accuracy, suggesting their usefulness in real-world HR applications, especially given the challenges even experts face in distinguishing between closely related experience levels.

#### [Salience and Market-aware Skill Extraction for Job Targeting](https://doi.org/10.1145/3394486.3403338) (Shi et al., 2020)

- **Authors:** Shi, Baoxu; Yang, Jaewon; Guo, Feng; He, Qi
- **Year:** 2020
- **Publication:** Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
- **Summary:** This paper introduces Job2Skills, a system designed to extract job-relevant skills by considering both their prominence in job postings and market trends such as supply and demand. Unlike traditional methods, Job2Skills evaluates the significance of skills dynamically, leading to better job recommendations and more accurate skill suggestions. Deployed on LinkedIn, it has improved job application rates and reduced skill suggestion rejections, benefiting millions of job postings globally.

#### [Learning Effective Representations for Person-Job Fit by Feature Fusion](https://doi.org/10.1145/3340531.3412717) (Jiang et al., 2020)

- **Authors:** Jiang, Junshu; Ye, Songyun; Wang, Wei; Xu, Jingran; Luo, Xiaosheng
- **Year:** 2020
- **Publication:** Proceedings of the 29th ACM International Conference on Information & Knowledge Management
- **Summary:** This paper proposes a method to improve person-job matching by learning better representations of candidates and job posts. It combines features from free text, extracted semantic entities, and historical application data to capture both explicit and implicit information. Experiments with real-world data demonstrate that this approach significantly outperforms existing methods and provides interpretable results.

#### [Learning to Match Jobs with Resumes from Sparse Interaction Data using Multi-View Co-Teaching Network](https://doi.org/10.1145/3340531.3411929) (Bian et al., 2020)

- **Authors:** Bian, Shuqing; Chen, Xu; Zhao, Wayne Xin; Zhou, Kun; Hou, Yupeng; Song, Yang; Zhang, Tao; Wen, Ji-Rong
- **Year:** 2020
- **Publication:** Proceedings of the 29th ACM International Conference on Information & Knowledge Management
- **Summary:** This paper proposes a novel multi-view co-teaching network to address challenges in job-resume matching caused by sparse and noisy interaction data on online recruitment platforms. The approach combines a text-based matching model and a relation-based matching model to capture semantic compatibility from complementary perspectives and employs two enhancement strategies: shared representation learning and a co-teaching mechanism that selects reliable training instances to mitigate noise. Experimental results show that this method outperforms state-of-the-art approaches, demonstrating its effectiveness in handling limited and noisy data for job-resume matching tasks.

#### [Implicit Skills Extraction Using Document Embedding and Its Use in Job Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/7038) (Gugnani et al., 2020)

- **Authors:** Gugnani, Akshay; Misra, Hemant
- **Year:** 2020
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper introduces a job recommender system that uses NLP techniques to extract both explicit and implicit skills from job descriptions and match them to candidates' resumes. Implicit skills, inferred based on context such as geography or industry, are identified using a semantic similarity approach leveraging a Doc2Vec model trained on over 1.1 million JDs. The system achieves significant performance improvements, with a mean reciprocal rank of 0.88, showcasing its efficacy compared to traditional explicit skill-based matching.

#### [Improving Next-Application Prediction with Deep Personalized-Attention Neural Network](https://ieeexplore.ieee.org/document/9680268) (Zhu et al., 2021)

- **Authors:** Zhu, Jun; Viaud, Gautier; Hudelot, Céline
- **Year:** 2021
- **Publication:** 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)
- **Summary:** This paper introduces a deep learning-based approach called Personalized-Attention Next-Application Prediction (PANAP) to improve job recommendation systems. PANAP uses three main modules to learn representations of job postings, job seekers, and predict relevant future applications. The model incorporates personalized attention and geographic factors, demonstrating its effectiveness through experiments on the CareerBuilder12 dataset.

#### [e-Recruitment recommender systems: a systematic review](https://doi.org/10.1007/s10115-020-01522-8) (Freire et al., 2021)

- **Authors:** Freire, Mauricio Noris; de Castro, Leandro Nunes
- **Year:** 2021
- **Publication:** Knowledge and Information Systems
- **Summary:** This survey paper reviews how recommender systems are applied to e-Recruitment, focusing on research published between 2012 and 2020. It categorizes 63 selected papers based on system types, information sources, and assessment methods, revealing a trend toward hybrid and innovative approaches. The study identifies key challenges and opportunities for improving RecSys in matching candidates with jobs.

#### [Knowledge Enhanced Person-Job Fit for Talent Recruitment](https://ieeexplore.ieee.org/document/9835552) (Yao et al., 2022)

- **Authors:** Yao, Kaichun; Zhang, Jingshuai; Qin, Chuan; Wang, Peng; Zhu, Hengshu; Xiong, Hui
- **Year:** 2022
- **Publication:** 2022 IEEE 38th International Conference on Data Engineering
- **Summary:** This paper proposes a method to improve person-job fit in talent recruitment by addressing the semantic gap between job postings and resumes. It introduces a skill extraction model and constructs a knowledge graph using skill entity dictionaries and unlabeled data. The method incorporates prior knowledge into a graph representation learning approach, which enhances the matching process between job postings and resumes.

#### [Modeling Two-Way Selection Preference for Person-Job Fit](https://dl.acm.org/doi/10.1145/3523227.3546752) (Yang et al., 2022)

- **Authors:** Yang, Chen; Hou, Yupeng; Song, Yang; Zhang, Tao; Wen, Ji-Rong; Zhao, Wayne Xin
- **Year:** 2022
- **Publication:** Proceedings of the 16th ACM Conference on Recommender Systems
- **Summary:** This paper proposes a method to model person-job fit by considering both candidates' and employers' perspectives in the recruitment process. It uses a dual-perspective graph representation to capture both successful and failed job matches. The approach is validated through experiments on large-scale datasets, demonstrating its effectiveness in improving recruitment efficiency.

#### [Machop: an end-to-end generalized entity matching framework](https://doi.org/10.1145/3533702.3534910) (Wang et al., 2022)

- **Authors:** Wang, Jin; Li, Yuliang; Hirota, Wataru; Kandogan, Eser
- **Year:** 2022
- **Publication:** Proceedings of the Fifth International Workshop on Exploiting Artificial Intelligence Techniques for Data Management
- **Summary:** This paper proposes a new approach to the Entity Matching problem, called Generalized Entity Matching (GEM), which addresses the challenges of matching diverse data formats and semantics. The authors introduce Machop, an end-to-end pipeline for solving GEM, utilizing Transformer-based LMs like BERT and incorporating a novel external knowledge injection method. The proposed method outperforms current approaches, achieving a 17.1% improvement in F1 score and providing matching results that are interpretable for human users.

#### [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://ceur-ws.org/Vol-2967/paper_8.pdf) (Lavi et al., 2022)

- **Authors:** Lavi, Dor; Medentsiy, Volodymyr; Graus, David
- **Year:** 2022
- **Publication:** Proceedings of the Workshop on Recommender Systems for Human Resources co-located with the 15th ACM Conference on Recommender Systems
- **Summary:** This paper presents a method for improving job-resume matching by fine-tuning BERT's next-sentence prediction task. Historical data on past matches and mismatches are used to train a Siamese network that compares job descriptions with resumes. Experiments on a Scandinavian job portal dataset demonstrate that this approach outperforms Sentence-BERT and other modern methods in assessing person-job fit.       

#### [Bilateral Sequential Hypergraph Convolution Network for Reciprocal Recommendation](https://ieeexplore.ieee.org/document/10415725) (Chen et al., 2023)

- **Authors:** Chen, Jiaxing; Liu, Hongzhi; Guo, Hongrui; Du, Yingpeng; Wang, Zekai; Song, Yang; Wu, Zhonghai
- **Year:** 2023
- **Publication:** 2023 IEEE International Conference on Data Mining
- **Summary:** This paper proposes a model for reciprocal recommendation using bilateral sequential hypergraphs. It aims to capture complex, multidimensional relationships among users and their evolving preferences over time. The model incorporates a novel convolution strategy that accounts for both sequential interactions and feedback behaviors, outperforming existing methods in various experiments.

#### [Fairness of recommender systems in the recruitment domain: an analysis from technical and legal perspectives](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2023.1245198) (Kumar et al., 2023)

- **Authors:** Kumar, Deepak; Grosz, Tessa; Rekabsaz, Navid; Greif, Elisabeth; Schedl, Markus
- **Year:** 2023
- **Publication:** Frontiers in Big Data
- **Summary:** This survey paper explores the fairness of recommender systems used in recruitment, considering both technical and legal aspects. It examines various fairness measures like demographic parity and equal opportunity, and evaluates methods such as synthetic data, adversarial training, and post-hoc re-ranking to improve fairness. Additionally, the paper reviews the alignment of these fairness strategies with legal frameworks in the EU and US, highlighting the challenges and limitations in implementing fair recruitment practices through automated systems.       

#### [An Exploration of Sentence-Pair Classification for Algorithmic Recruiting](https://doi.org/10.1145/3604915.3610657) (Kaya et al., 2023)

- **Authors:** Kaya, Mesut; Bogers, Toine
- **Year:** 2023
- **Publication:** Proceedings of the 17th ACM Conference on Recommender Systems
- **Summary:** This paper explores a novel approach to improving job-resume matching using a fine-tuned BERT model. The method adapts BERT's next-sentence prediction task to estimate how well a resume fits a job description, utilizing past data on successful and unsuccessful matches. Experiments reveal that this technique outperforms Sentence-BERT and other advanced methods in predicting person-job compatibility.

#### [Disentangling and Operationalizing AI Fairness at LinkedIn](https://doi.org/10.1145/3593013.3594075) (Quiñonero Candela et al., 2023)

- **Authors:** Quiñonero Candela, Joaquin; Wu, Yuwen; Hsu, Brian; Jain, Sakshi; Ramos, Jennifer; Adams, Jon; Hallman, Robert; Basu, Kinjal
- **Year:** 2023
- **Publication:** Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper addresses the challenges of implementing AI fairness at LinkedIn’s scale, particularly with the variety of fairness definitions and the need for context-specific fairness. It introduces a framework that separates equal treatment and equitable product expectations, offering guidelines to ensure fair AI practices while balancing these elements.

#### [BOSS: A Bilateral Occupational-Suitability-Aware Recommender System for Online Recruitment](https://doi.org/10.1145/3580305.3599783) (Hu et al., 2023)

- **Authors:** Hu, Xiao; Cheng, Yuan; Zheng, Zhi; Wang, Yue; Chi, Xinxin; Zhu, Hengshu
- **Year:** 2023
- **Publication:** Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining
- **Summary:** This paper presents BOSS, a recommender system for online recruitment that considers the preferences of both job seekers and recruiters. The proposed method uses a mixture-of-experts approach to learn these preferences and a multi-task learning framework to model the recruitment process. Experimental results from real-world datasets and online testing show that BOSS outperforms existing systems.

#### [A Challenge-based Survey of E-recruitment Recommendation Systems](https://doi.org/10.1145/3659942) (Mashayekhi et al., 2024)

- **Authors:** Mashayekhi, Yoosof; Li, Nan; Kang, Bo; Lijffijt, Jefrey; De Bie, Tijl
- **Year:** 2024
- **Publication:** ACM Computing Surveys
- **Summary:** This survey paper reviews the challenges faced by e-recruitment recommendation systems and explores how these challenges have been addressed in existing literature. Unlike previous surveys that focus on algorithmic methods, this paper takes a challenge-based approach, offering practical insights for developers and researchers in the field. The paper concludes by suggesting future research directions that may lead to significant improvements in e-recruitment systems.

#### [Fairness and Bias in Algorithmic Hiring: a Multidisciplinary Survey](https://dl.acm.org/doi/10.1145/3696457) (Fabris et al., 2024)

- **Authors:** Fabris, Alessandro; Baranowska, Nina; Dennis, Matthew J.; Graus, David; Hacker, Philipp; Saldivar, Jorge; Borgesius, Frederik Zuiderveen; Biega, Asia J.
- **Year:** 2024
- **Publication:** ACM Transactions on Intelligent Systems and Technology
- **Summary:** This survey paper reviews the current state of algorithmic hiring systems, focusing on fairness and bias. It explores the various challenges and opportunities in the use of these systems, considering their impact on different stakeholders. The survey also provides guidance on how to manage and govern algorithmic hiring to promote fairness and reduce bias, with a focus on future improvements.

#### [Identifying and Improving Disability Bias in GPT-Based Resume Screening](https://doi.org/10.1145/3630106.3658933) (Glazko et al., 2024)

- **Authors:** Glazko, Kate; Mohammed, Yusuf; Kosa, Ben; Potluri, Venkatesh; Mankoff, Jennifer
- **Year:** 2024
- **Publication:** Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper analyzes the presence of disability bias in GPT-4’s resume evaluations. By comparing standard resumes with versions containing disability-related achievements, the study reveals noticeable biases against applicants with these additions. The research demonstrates that custom training focused on diversity, equity, and inclusion (DEI) principles can significantly reduce this bias, pointing to potential solutions for fairer AI-based hiring practices.

#### [Do Large Language Models Discriminate in Hiring Decisions on the Basis of Race, Ethnicity, and Gender?](https://aclanthology.org/2024.acl-short.37/) (An et al., 2024)

- **Authors:** An, Haozhe; Acquaye, Christabel; Wang, Colin; Li, Zongxia; Rudinger, Rachel
- **Year:** 2024
- **Publication:** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
- **Summary:** This work analyzes whether LLMs show bias in hiring decisions based on the perceived race, ethnicity, and gender of applicants. Using templatic prompts that manipulate the applicant's name, the study finds that LLMs are more likely to favor White applicants over Hispanic ones, with masculine White names having the highest acceptance rates and masculine Hispanic names the lowest. The study also suggests that these biases vary depending on the structure of the prompts, indicating that LLMs' sensitivity to race and gender can differ across contexts.       

#### [Enhancing Job Recommendation through LLM-Based Generative Adversarial Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28678) (Du et al., 2024)

- **Authors:** Du, Yingpeng; Luo, Di; Yan, Rui; Wang, Xiaopei; Liu, Hongzhi; Zhu, Hengshu; Song, Yang; Zhang, Jie
- **Year:** 2024
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper proposes a method for improving job recommendations by using LLMs in combination with GANs. The approach focuses on enriching users' resumes by extracting both explicit and implicit information to overcome limitations like fabricated generation and few-shot problems. Experiments show that this method significantly improves the quality of job recommendations compared to traditional techniques.

#### [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations](https://ojs.aaai.org/index.php/AAAI/article/view/28769) (Wu et al., 2024)

- **Authors:** Wu, Likang; Qiu, Zhaopeng; Zheng, Zhi; Zhu, Hengshu; Chen, Enhong
- **Year:** 2024
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This studies explores the use of LLMs for understanding behavior graphs in online job recommendation systems. It introduces a novel framework that leverages LLMs to analyze behavior graphs, offering personalized and accurate job suggestions, including for out-of-distribution applications. The proposed method demonstrates improved recommendation quality on real-world datasets.

#### [Fake Resume Attacks: Data Poisoning on Online Job Platforms](https://doi.org/10.1145/3589334.3645524) (Yamashita et al., 2024)

- **Authors:** Yamashita, Michiharu; Tran, Thanh; Lee, Dongwon
- **Year:** 2024
- **Publication:** Proceedings of the ACM on Web Conference 2024
- **Summary:** This paper explores critical vulnerabilities in online job platforms, focusing on how data poisoning attacks affect the matching of job seekers with companies. This study identifies three types of attacks: promoting certain companies, demoting others, and increasing the chances of specific users being matched with specific companies. By using a framework named FRANCIS to create fake resumes, the study demonstrates that such attacks can significantly distort matchmaking outcomes.

#### [JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer](https://dl.acm.org/doi/10.1145/3701735) (JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer, 2024)

- **Authors:** JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer
- **Year:** 2024
- **Publication:** ACM Transactions on Knowledge Discovery from Data
- **Summary:** This paper proposes a skill-aware job recommendation system that uses a semantic-enhanced Transformer model to bridge the gap between job descriptions (JDs) and user profiles. It addresses challenges like the lack of detailed information in JDs and differences between JD content and user skills by leveraging a two-stage learning approach. The proposed model improves job recommendation accuracy by incorporating both local and global attention mechanisms to capture dependencies within and across job-related data, showing promising results on large-scale datasets.

### Training And Learning

#### [Recommending Courses in MOOCs for Jobs: An Auto Weak Supervision Approach](https://doi.org/10.1007/978-3-030-67667-4_3) (Hao et al., 2021)

- **Authors:** Hao, Bowen; Zhang, Jing; Li, Cuiping; Chen, Hong; Yin, Hongzhi
- **Year:** 2021
- **Publication:** Machine Learning and Knowledge Discovery in Databases: Applied Data Science Track
- **Summary:** This paper presents a novel framework called AutoWeakS, which leverages reinforcement learning to address the challenge of recommending MOOCs for job seekers. AutoWeakS trains supervised ranking models using pseudo-labels generated by multiple unsupervised models and optimally combines these models to improve performance. Experimental evaluations across various job and course datasets demonstrate that AutoWeakS significantly surpasses traditional unsupervised, supervised, and weakly supervised approaches.

#### [Course Recommender Systems Need to Consider the Job Market](https://dl.acm.org/doi/abs/10.1145/3626772.3657847) (Frej et al., 2024)

- **Authors:** Frej, Jibril; Dai, Anna; Montariol, Syrielle; Bosselut, Antoine; Käser, Tanja
- **Year:** 2024
- **Publication:** Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieva
- **Summary:** This work focuses on the development of course recommender systems that incorporate job market skill demands, highlighting the need for integration between current systems and labor market evolutions. It discusses essential properties for such systems, including the need for explainability, sequence handling, unsupervised learning, and alignment with both the job market and users’ personal goals. The study introduces an initial system leveraging large language models for skill extraction and reinforcement learning for job market alignment, providing empirical evidence of its effectiveness using open-source data.

#### [Generative Learning Plan Recommendation for Employees: A Performance-aware Reinforcement Learning Approach](https://dl.acm.org/doi/abs/10.1145/3604915.3608795) (Zheng et al., 2023)

- **Authors:** Zheng, Zhi; Sun, Ying; Song, Xin; Zhu, Hengshu; Xiong, Hui
- **Year:** 2023
- **Publication:** Recsys 2023
- **Summary:** This paper introduces the Generative Learning plAn recommenDation (GLAD) framework to create personalized learning plans for employees that aim to enhance their work performance. Unlike existing approaches that prioritize user preferences, GLAD integrates a performance predictor and a rationality discriminator—both transformer-based models—to ensure recommendations are both beneficial to career development and logically sequenced. Experimental validation on real-world data demonstrates GLAD's effectiveness compared to baseline methods, offering new insights for talent management.

#### [Collaboration-Aware Hybrid Learning for Knowledge Development Prediction](https://dl.acm.org/doi/abs/10.1145/3589334.3645326) (Chen et al., 2024)

- **Authors:** Chen, Liyi; Qin, Chuan; Sun, Ying; Song, Xin; Xu, Tong; Zhu, Hengshu; Xiong, Hui
- **Year:** 2024
- **Publication:** Proceedings of the ACM Web Conference 2024
- **Summary:** This study proposes a Collaboration-Aware Hybrid Learning (CAHL) approach for predicting employees' future knowledge acquisition and assessing the impact of collaborative learning patterns. The method uses a Job Knowledge Embedding module to learn co-occurrence and prerequisite relationships of knowledge, and an Employee Embedding module to aggregate information about employees and their work collaborators. Through the Hybrid Learning Simulation module, the model integrates collaborative and self-learning to effectively predict future job knowledge development, with experiments validating its effectiveness.

#### [Personalized and Explainable Employee Training Course Recommendations: A Bayesian Variational Approach](nan) (Wang et al., 2021)

- **Authors:** Wang, Chao; Zhu, Hengshu; Wang, Peng; Zhu, Chen; Zhang, Xi; Chen, Enhong; Xiong, Hui
- **Year:** 2021
- **Publication:** ACM Transactions on Information Systems
- **Summary:** This paper introduces a Bayesian variational framework, DCBVN, for personalized and explainable employee training recommendations by modeling employees’ current skills and career development goals. It uses autoencoding variational inference to extract interpretable competency representations and employs a demand recognition mechanism to understand career development needs. To address sparse data, an enhanced version (DCCAN) integrates graph-attentive networks, leveraging connections among employees to infer competencies and deliver robust, explainable recommendations.

### Skill Recommendation

#### [A Combined Representation Learning Approach for Better Job and Skill Recommendation](https://dl.acm.org/doi/10.1145/3269206.3272023) (Dave et al., 2018)

- **Authors:** Dave, Vachik S.; Zhang, Baichuan; Al Hasan, Mohammad; AlJadda, Khalifeh; Korayem, Mohammed
- **Year:** 2018
- **Publication:** Proceedings of the 27th ACM International Conference on Information and Knowledge Management
- **Summary:** This paper proposes a novel job and skill recommendation system by utilizing three types of information networks: job transition, job-skill, and skill co-occurrence networks. The authors introduce a representation learning model that jointly learns the representations of jobs and skills in a shared latent space, improving recommendations for both jobs and the skills needed for them. Experiments and case studies demonstrate that their model outperforms traditional methods, offering more effective job and skill suggestions for users.

#### [Measuring the Popularity of Job Skills in Recruitment Market: A Multi-Criteria Approach](https://aaai.org/papers/11847-measuring-the-popularity-of-job-skills-in-recruitment-market-a-multi-criteria-approach/) (Xu et al., 2018)

- **Authors:** Xu, Tong; Zhu, Hengshu; Zhu, Chen; Li, Pan; Xiong, Hui
- **Year:** 2018
- **Publication:** Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence
- **Summary:** This paper introduces a data-driven approach to measure the popularity of job skills by analyzing large-scale recruitment data and constructing a job skill network. A novel Skill Popularity-based Topic Model (SPTM) is proposed, integrating diverse job criteria (e.g., salary levels, company size) and the latent relationships among skills to rank them based on multifaceted popularity. Experiments on real-world recruitment data demonstrate the effectiveness of SPTM, uncovering trends like popular skills associated with high-paying jobs.

### Job Posting Generation

#### [Hiring Now: A Skill-Aware Multi-Attention Model for Job Posting Generation](https://aclanthology.org/2020.acl-main.281/) (Liu et al., 2020)

- **Authors:** Liu, Liting; Liu, Jie; Zhang, Wenzheng; Chi, Ziming; Shi, Wenxuan; Huang, Yalou
- **Year:** 2020
- **Publication:** Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics
- **Summary:** This paper introduces a novel approach to Job Posting Generation by framing it as a conditional text generation problem that creates job requirements based on job descriptions. The proposed model, SAMA (Skill-Aware Multi-Attention), uses a hierarchical decoder to identify key skills from job descriptions and generates postings guided by these skill labels. Additionally, a skill knowledge graph enriches the generation process by incorporating meta information like job titles and company sizes, leading to more accurate and comprehensive job postings, as demonstrated by experiments on real-world data.

#### [Transfer Learning for Multilingual Vacancy Text Generation](https://aclanthology.org/2022.gem-1.18) (Lorincz et al., 2022)

- **Authors:** Lorincz, Anna; Graus, David; Lavi, Dor; Pereira, Joao Lebre Magalhaes
- **Year:** 2022
- **Publication:** Proceedings of the 2nd Workshop on Natural Language Generation, Evaluation, and Metrics (GEM)
- **Summary:** This work uses mT5 to automatically generate benefit sections of job vacancy texts in multiple languages. A new domain-specific evaluation metric was developed to measure how accurately the generated text includes structured input data like binary, categorical, and numeric information. Results show that while mT5 performs well with categorical and binary inputs, it struggles with unseen city names and numeric data, although accuracy can improve with synthetic training data for integer-based numeric inputs.

#### [Looking for a Handsome Carpenter! Debiasing GPT-3 Job Advertisements](https://aclanthology.org/2022.gebnlp-1.22) (Borchers et al., 2022)

- **Authors:** Borchers, Conrad; Gala, Dalia; Gilburt, Benjamin; Oravkin, Eduard; Bounsi, Wilfried; Asano, Yuki M; Kirk, Hannah
- **Year:** 2022
- **Publication:** Proceedings of the 4th Workshop on Gender Bias in Natural Language Processing
- **Summary:** This paper analyzes the use of GPT-3 to create unbiased job advertisements. It compares the quality and fairness of GPT-3’s zero-shot outputs with real-world ads, exploring both prompt-engineering and fine-tuning as methods to reduce bias. The results show that while prompt-engineering has limited effect, fine-tuning with unbiased data improves both realism and fairness significantly.

#### [Towards Automatic Job Description Generation With Capability-Aware Neural Networks](https://ieeexplore.ieee.org/document/9693259) (Qin et al., 2023)

- **Authors:** Qin, Chuan; Yao, Kaichun; Zhu, Hengshu; Xu, Tong; Shen, Dazhong; Chen, Enhong; Xiong, Hui
- **Year:** 2023
- **Publication:** IEEE Transactions on Knowledge and Data Engineering
- **Summary:** This paper proposes a method for generating job descriptions using a neural network framework called Cajon. Cajon integrates a capability-aware neural topic model to extract relevant capability information from large recruitment datasets and uses an encoder-decoder architecture for generating job descriptions. The framework aims to reduce human effort in creating job descriptions by ensuring they are comprehensive, accurate, and tailored to specific job requirements.

### Resume Writing

#### [LinkedIn Skills: Large-Scale Topic Extraction and Inference](https://doi.org/10.1145/2645710.2645729) (Bastian et al., 2014)

- **Authors:** Bastian, Mathieu; Hayes, Matthew; Vaughan, William; Shah, Sam; Skomoroch, Peter; Kim, Hyungjin; Uryasev, Sal; Lloyd, Christopher
- **Year:** 2014
- **Publication:** Proceedings of the 8th ACM Conference on Recommender systems
- **Summary:** This paper presents the development of a large-scale IE pipeline for LinkedIn's "Skills and Expertise" feature. The pipeline includes constructing a folksonomy of skills and expertise and implementing an inference and recommender system for skills. The paper also discusses applications like Endorsements, which allows members to tag themselves with topics representing their areas of expertise and for their connections to provide social proof of that member's competence in that topic.

### Reference Letter Generation

#### [“Kelly is a Warm Person, Joseph is a Role Model”: Gender Biases in LLM-Generated Reference Letters](https://aclanthology.org/2023.findings-emnlp.243) (Wan et al., 2023)

- **Authors:** Wan, Yixin; Pu, George; Sun, Jiao; Garimella, Aparna; Chang, Kai-Wei; Peng, Nanyun
- **Year:** 2023
- **Publication:** Findings of the Association for Computational Linguistics: EMNLP 2023
- **Summary:** This paper analyzes gender biases in reference letters generated by LLMs. It focuses on two main areas of bias: language style and lexical content. The study finds significant gender biases in LLM-generated letters, emphasizing the need for careful review of LLM outputs to avoid potential harms, particularly to female applicants.

### Fairness And Bias

#### [Looking for a Handsome Carpenter! Debiasing GPT-3 Job Advertisements](https://aclanthology.org/2022.gebnlp-1.22) (Borchers et al., 2022)

- **Authors:** Borchers, Conrad; Gala, Dalia; Gilburt, Benjamin; Oravkin, Eduard; Bounsi, Wilfried; Asano, Yuki M; Kirk, Hannah
- **Year:** 2022
- **Publication:** Proceedings of the 4th Workshop on Gender Bias in Natural Language Processing
- **Summary:** This paper analyzes the use of GPT-3 to create unbiased job advertisements. It compares the quality and fairness of GPT-3’s zero-shot outputs with real-world ads, exploring both prompt-engineering and fine-tuning as methods to reduce bias. The results show that while prompt-engineering has limited effect, fine-tuning with unbiased data improves both realism and fairness significantly.

#### [Is a Prestigious Job the same as a Prestigious Country? A Case Study on Multilingual Sentence Embeddings and European Countries](https://aclanthology.org/2023.findings-emnlp.71) (Libovický, 2023)

- **Authors:** Libovický, Jindřich
- **Year:** 2023
- **Publication:** Findings of the Association for Computational Linguistics: EMNLP 2023
- **Summary:** This paper explores how multilingual sentence embeddings represent countries and occupations, focusing on European nations. It shows that embeddings mostly reflect the economic strength and political divide between Eastern and Western Europe, with job prestige being clearly separated in the data. The study finds that job prestige does not correlate with the country dimensions in most models, except for one smaller model where a potential link between job prestige and nationality is observed.       

#### [Fairness of recommender systems in the recruitment domain: an analysis from technical and legal perspectives](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2023.1245198) (Kumar et al., 2023)

- **Authors:** Kumar, Deepak; Grosz, Tessa; Rekabsaz, Navid; Greif, Elisabeth; Schedl, Markus
- **Year:** 2023
- **Publication:** Frontiers in Big Data
- **Summary:** This survey paper explores the fairness of recommender systems used in recruitment, considering both technical and legal aspects. It examines various fairness measures like demographic parity and equal opportunity, and evaluates methods such as synthetic data, adversarial training, and post-hoc re-ranking to improve fairness. Additionally, the paper reviews the alignment of these fairness strategies with legal frameworks in the EU and US, highlighting the challenges and limitations in implementing fair recruitment practices through automated systems.       

#### [Disentangling and Operationalizing AI Fairness at LinkedIn](https://doi.org/10.1145/3593013.3594075) (Quiñonero Candela et al., 2023)

- **Authors:** Quiñonero Candela, Joaquin; Wu, Yuwen; Hsu, Brian; Jain, Sakshi; Ramos, Jennifer; Adams, Jon; Hallman, Robert; Basu, Kinjal
- **Year:** 2023
- **Publication:** Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper addresses the challenges of implementing AI fairness at LinkedIn’s scale, particularly with the variety of fairness definitions and the need for context-specific fairness. It introduces a framework that separates equal treatment and equitable product expectations, offering guidelines to ensure fair AI practices while balancing these elements.

#### [Identifying and Improving Disability Bias in GPT-Based Resume Screening](https://doi.org/10.1145/3630106.3658933) (Glazko et al., 2024)

- **Authors:** Glazko, Kate; Mohammed, Yusuf; Kosa, Ben; Potluri, Venkatesh; Mankoff, Jennifer
- **Year:** 2024
- **Publication:** Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper analyzes the presence of disability bias in GPT-4’s resume evaluations. By comparing standard resumes with versions containing disability-related achievements, the study reveals noticeable biases against applicants with these additions. The research demonstrates that custom training focused on diversity, equity, and inclusion (DEI) principles can significantly reduce this bias, pointing to potential solutions for fairer AI-based hiring practices.

### Job Interview

#### [A Job Interview Simulation: Social Cue-Based Interaction with a Virtual Character](https://ieeexplore.ieee.org/document/6693336?denied=) (Baur et al., 2013)

- **Authors:** Baur, Tobias; Damian, Ionut; Gebhard, Patrick; Porayska-Pomsta, Kaska; André, Elisabeth
- **Year:** 2013
- **Publication:** 2013 International Conference on Social Computing
- **Summary:** This paper presents a virtual character that plays the role of a recruiter in a job interview simulation environment. This virtual character is designed to react and adapt to the user's behavior through the automatic recognition of social cues (conscious or unconscious behavioral patterns).

#### [MACH: My Automated Conversation Coach](https://dl.acm.org/doi/10.1145/2493432.2493502) (Hoque et al., 2014)

- **Authors:** Hoque, Mohammed (Ehsan); Courgeon, Matthieu; Martin, Jean-Claude; Mutlu, Bilge; Picard, Rosalind W.
- **Year:** 2014
- **Publication:** Proceedings of the 2013 ACM International Joint Conference on Pervasive and Ubiquitous Computing
- **Summary:** This paper introduces MACH, a virtual coach designed to help users improve social skills, specifically for job interviews. The system uses a virtual agent that recognizes and responds to facial expressions and speech, offering real-time feedback. In trials with MIT students, those who practiced with MACH showed measurable improvement in interview performance compared to a control group.

#### [Automated Analysis and Prediction of Job Interview Performance](https://ieeexplore.ieee.org/document/7579163) (Naim et al., 2018)

- **Authors:** Naim, Iftekhar; Tanveer, Md. Iftekhar; Gildea, Daniel; Hoque, Mohammed Ehsan
- **Year:** 2018
- **Publication:** IEEE Transactions on Affective Computing
- **Summary:** This paper proposes a system that uses video analysis to assess job interview performance by examining verbal and nonverbal behaviors, such as speech patterns and facial expressions. The framework, tested on interviews with MIT students, predicts interview traits like engagement and friendliness with high accuracy. It also suggests practical improvements for interviewees, such as reducing filler words and smiling more, emphasizing the importance of a strong first impression.

#### [Automatic assessment of communication skill in non-conventional interview settings: a comparative study](https://dl.acm.org/doi/10.1145/3136755.3136756) (Rao S. B et al., 2018)

- **Authors:** Rao S. B, Pooja; Rasipuram, Sowmya; Das, Rahul; Jayagopi, Dinesh Babu
- **Year:** 2018
- **Publication:** Proceedings of the 19th ACM International Conference on Multimodal Interaction
- **Summary:** This study compares the effectiveness of assessing communication skills in two non-conventional interview formats: asynchronous video interviews and written interviews. It introduces a predictive model that uses audio, visual, and textual features to evaluate candidates, achieving 75% accuracy in binary classification. The study also explores how automated predictions align with human expert evaluations across different interview settings.

#### [Spoken Dialogue System for a Human-like Conversational Robot ERICA](https://doi.org/10.1007/978-981-13-9443-0_6) (Kawahara, 2019)

- **Authors:** Kawahara, Tatsuya
- **Year:** 2019
- **Publication:** 9th International Workshop on Spoken Dialogue System Technology
- **Summary:** This paper describes the development of ERICA, a conversational android designed for human-like interactions in roles such as counseling and job interviews. The focus is on creating a natural dialogue system, especially for attentive listening, incorporating features like backchannels, fillers, and laughter to mimic human conversation.

#### [Serious Games for Training Social Skills in Job Interviews](https://ieeexplore.ieee.org/document/8299545) (Gebhard et al., 2019)

- **Authors:** Gebhard, Patrick; Schneeberger, Tanja; André, Elisabeth; Baur, Tobias; Damian, Ionut; Mehlmann, Gregor; König, Cornelius; Langer, Markus
- **Year:** 2019
- **Publication:** IEEE Transactions on Games
- **Summary:** This study explores a serious game designed to train social skills for job interviews using virtual agents. The game includes scenario-based role plays where users interact with lifelike characters in a 3D environment, simulating real interview situations. Studies show how different agent personalities and environments influence the user's perception and learning outcomes.

#### [Virtual Job Interviewing Practice for High-Anxiety Populations](https://dl.acm.org/doi/10.1145/3308532.3329417) (Hartholt et al., 2019)

- **Authors:** Hartholt, Arno; Mozgai, Sharon; Rizzo, Albert "Skip"
- **Year:** 2019
- **Publication:** Proceedings of the 19th ACM International Conference on Intelligent Virtual Agents
- **Summary:** This paper introduces a system designed to help individuals facing significant job interview challenges. It features customizable virtual interviewers who simulate various conversation styles and job environments. The system aims to build confidence and adaptability, particularly for those with Autism Spectrum Disorder, veterans, and formerly incarcerated individuals.

#### [DuerQuiz: A Personalized Question Recommender System for Intelligent Job Interview](https://doi.org/10.1145/3292500.3330706) (Qin et al., 2019)

- **Authors:** Qin, Chuan; Zhu, Hengshu; Zhu, Chen; Xu, Tong; Zhuang, Fuzhen; Ma, Chao; Zhang, Jingshuai; Xiong, Hui
- **Year:** 2019
- **Publication:** Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
- **Summary:** This work presents DuerQuiz, a personalized question recommender system designed to enhance job interview assessments by generating relevant questions based on a knowledge graph of job skills (Skill-Graph). The system utilizes a bidirectional LSTM-CRF model with an adapted gate mechanism for skill entity extraction, improved by label propagation from large-scale query data, and constructs hypernym-hyponym relations between skills. DuerQuiz's personalized question recommendation algorithm, validated through real-world recruitment data, significantly improves the efficiency and effectiveness of the interview process, as demonstrated during the 2018 Baidu campus recruitment event.

#### [HireNet: A Hierarchical Attention Model for the Automatic Analysis of Asynchronous Video Job Interviews](https://ojs.aaai.org/index.php/AAAI/article/view/3832) (Hemamou et al., 2019)

- **Authors:** Hemamou, Léo; Felhi, Ghazi; Vandenbussche, Vincent; Martin, Jean-Claude; Clavel, Chloé
- **Year:** 2019
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper introduces HireNet, a hierarchical attention model designed to predict a candidate's hirability in asynchronous video interviews. Using a dataset of over 7,000 real interview recordings, HireNet analyzes both the verbal content and social signals in the responses. The model outperforms previous methods in accuracy and offers insights into key moments that may influence hiring decisions.

#### [Job Interviewer Android with Elaborate Follow-up Question Generation](https://doi.org/10.1145/3382507.3418839) (Inoue et al., 2020)

- **Authors:** Inoue, Koji; Hara, Kohei; Lala, Divesh; Yamamoto, Kenta; Nakamura, Shizuka; Takanashi, Katsuya; Kawahara, Tatsuya
- **Year:** 2020
- **Publication:** Proceedings of the 2020 International Conference on Multimodal Interaction
- **Summary:** This paper presents a system where an embodied robot conducts job interviews by generating dynamic follow-up questions based on the interviewee's responses. The system was compared to a baseline using fixed questions and showed superior performance in terms of question quality and overall interview experience. Similar positive results were observed with a virtual agent, though the sense of presence was enhanced only with the (embodied) android interviewer.

#### [Automatic Follow-up Question Generation for Asynchronous Interviews](https://aclanthology.org/2020.intellang-1.2) (Rao S B et al., 2020)

- **Authors:** Rao S B, Pooja; Agnihotri, Manish; Jayagopi, Dinesh Babu
- **Year:** 2020
- **Publication:** Proceedings of the Workshop on Intelligent Information Processing and Natural Language Generation
- **Summary:** This paper proposes a system that improves asynchronous video interviews by generating relevant follow-up questions. The model, integrated into a 3D virtual interviewer called Maya, creates natural and diverse questions based on prior responses. Human evaluations show that the system's generated questions are 77% relevant, surpassing traditional scripted approaches and baseline models.

#### [Generating An Optimal Interview Question Plan Using A Knowledge Graph And Integer Linear Programming](https://aclanthology.org/2021.naacl-main.160/) (Datta et al., 2021)

- **Authors:** Datta, Soham; Mallick, Prabir; Patil, Sangameshwar; Bhattacharya, Indrajit; Palshikar, Girish
- **Year:** 2021
- **Publication:** Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
- **Summary:** This paper presents an interview assistant system designed to select personalized technical questions for job candidates. The system uses integer linear programming and a knowledge graph to generate an optimal question set based on the candidate's resume. Evaluation shows the system's effectiveness by comparing its suggested questions to real interview questions and expert feedback.

#### [Simultaneous Job Interview System Using Multiple Semi-autonomous Agents](https://aclanthology.org/2022.sigdial-1.12) (Kawai et al., 2022)

- **Authors:** Kawai, Haruki; Muraki, Yusuke; Yamamoto, Kenta; Lala, Divesh; Inoue, Koji; Kawahara, Tatsuya
- **Year:** 2022
- **Publication:** Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue
- **Summary:** This paper proposes a system where one interviewer can conduct simultaneous one-on-one job interviews with multiple applicants using semi-autonomous dialogue systems. The system evaluates applicant responses and extracts key points to help the interviewer understand and manage parallel interviews. A pilot experiment shows that this approach can assist interviewers in smoothly guiding conversations when needed.

#### [“Am I Answering My Job Interview Questions Right?”: A NLP Approach to Predict Degree of Explanation in Job Interview Responses](https://aclanthology.org/2022.nlp4pi-1.14) (Verrap et al., 2022)

- **Authors:** Verrap, Raghu; Nirjhar, Ehsanul; Nenkova, Ani; Chaspari, Theodora
- **Year:** 2022
- **Publication:** Proceedings of the Second Workshop on NLP for Positive Impact
- **Summary:** This study explores NLP methods to evaluate the level of explanation in job interview responses. By analyzing mock interviews with military veterans, it identifies effective techniques for detecting under- and over-explained answers. The findings support developing AI tools to assist job candidates in improving their interview skills, fostering a more inclusive workforce.

#### [EZInterviewer: To Improve Job Interview Performance with Mock Interview Generator](https://doi.org/10.1145/3539597.3570476) (Li et al., 2023)

- **Authors:** Li, Mingzhe; Chen, Xiuying; Liao, Weiheng; Song, Yang; Zhang, Tao; Zhao, Dongyan; Yan, Rui
- **Year:** 2023
- **Publication:** Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining
- **Summary:** This paper proposes EZInterviewer, a mock interview generator based on a limtied amount of real interview data. The system addresses the challenge of low-resource data by separating knowledge selection from dialog generation, training its modules on resumes and ungrounded dialogs. Evaluation on real interview data shows that EZInterviewer effectively generates meaningful mock interviews, offering job seekers a more authentic practice experience.

### Job Interview Assessment

#### [Automated Analysis and Prediction of Job Interview Performance](https://ieeexplore.ieee.org/document/7579163) (Naim et al., 2018)

- **Authors:** Naim, Iftekhar; Tanveer, Md. Iftekhar; Gildea, Daniel; Hoque, Mohammed Ehsan
- **Year:** 2018
- **Publication:** IEEE Transactions on Affective Computing
- **Summary:** This paper proposes a system that uses video analysis to assess job interview performance by examining verbal and nonverbal behaviors, such as speech patterns and facial expressions. The framework, tested on interviews with MIT students, predicts interview traits like engagement and friendliness with high accuracy. It also suggests practical improvements for interviewees, such as reducing filler words and smiling more, emphasizing the importance of a strong first impression.

#### [HireNet: A Hierarchical Attention Model for the Automatic Analysis of Asynchronous Video Job Interviews](https://ojs.aaai.org/index.php/AAAI/article/view/3832) (Hemamou et al., 2019)

- **Authors:** Hemamou, Léo; Felhi, Ghazi; Vandenbussche, Vincent; Martin, Jean-Claude; Clavel, Chloé
- **Year:** 2019
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper introduces HireNet, a hierarchical attention model designed to predict a candidate's hirability in asynchronous video interviews. Using a dataset of over 7,000 real interview recordings, HireNet analyzes both the verbal content and social signals in the responses. The model outperforms previous methods in accuracy and offers insights into key moments that may influence hiring decisions.

#### [Joint Representation Learning with Relation-Enhanced Topic Models for Intelligent Job Interview Assessment](https://dl.acm.org/doi/10.1145/3469654) (Shen et al., 2021)

- **Authors:** Shen, Dazhong; Qin, Chuan; Zhu, Hengshu; Xu, Tong; Chen, Enhong; Xiong, Hui
- **Year:** 2021
- **Publication:** ACM Transactions on Information Systems
- **Summary:** This paper introduces three models to enhance the job interview assessment process by learning from large-scale, real-world interview data. The models—JLMIA, Neural-JLMIA, and Refined-JLMIA—focus on mining relationships between job descriptions, resumes, and assessments while capturing core competencies and semantic evolution over multiple interview rounds. Experimental results show that these models reduce bias and improve the fairness and transparency of interview evaluations.

#### [“Am I Answering My Job Interview Questions Right?”: A NLP Approach to Predict Degree of Explanation in Job Interview Responses](https://aclanthology.org/2022.nlp4pi-1.14) (Verrap et al., 2022)

- **Authors:** Verrap, Raghu; Nirjhar, Ehsanul; Nenkova, Ani; Chaspari, Theodora
- **Year:** 2022
- **Publication:** Proceedings of the Second Workshop on NLP for Positive Impact
- **Summary:** This study explores NLP methods to evaluate the level of explanation in job interview responses. By analyzing mock interviews with military veterans, it identifies effective techniques for detecting under- and over-explained answers. The findings support developing AI tools to assist job candidates in improving their interview skills, fostering a more inclusive workforce.

### Job Interview Question Generation

#### [Generating An Optimal Interview Question Plan Using A Knowledge Graph And Integer Linear Programming](https://aclanthology.org/2021.naacl-main.160/) (Datta et al., 2021)

- **Authors:** Datta, Soham; Mallick, Prabir; Patil, Sangameshwar; Bhattacharya, Indrajit; Palshikar, Girish
- **Year:** 2021
- **Publication:** Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
- **Summary:** This paper presents an interview assistant system designed to select personalized technical questions for job candidates. The system uses integer linear programming and a knowledge graph to generate an optimal question set based on the candidate's resume. Evaluation shows the system's effectiveness by comparing its suggested questions to real interview questions and expert feedback.

### Screening Question Generation

#### [Learning to Ask Screening Questions for Job Postings](https://dl.acm.org/doi/10.1145/3397271.3401118) (Shi et al., 2020)

- **Authors:** Shi, Baoxu; Li, Shan; Yang, Jaewon; Kazdagli, Mustafa Emre; He, Qi
- **Year:** 2020
- **Publication:** Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval
- **Summary:** This paper introduces Job2Questions, a system to automatically generate screening questions for job postings on LinkedIn. It uses a two-stage deep learning model to detect and rank important intents from job descriptions, helping recruiters filter qualified candidates more effectively. The system significantly improves hiring efficiency, resulting in increased recruiter-applicant interactions and positive feedback from users.

### Labor Market Analysis

#### [Occupational profiling driven by online job advertisements: Taking the data analysis and processing engineering technicians as an example](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253308) (Cao et al., 2021)

- **Authors:** Cao, Lina; Zhang, Jian; Ge, Xinquan; Chen, Jindong
- **Year:** 2021
- **Publication:** PLOS ONE
- **Summary:** This paper demonstrates a method to enhance occupational profiling by using online job advertisements, focusing specifically on data analysis and processing engineering technicians (DAPET). The approach involves a multi-step process: first, identifying relevant job ads using text similarity algorithms; second, employing TextCNN for precise classification; and third, extracting named entities related to specialties and skills. The resulting occupation characteristics create a dynamic and detailed vocational profile that addresses the limitations of traditional profiling systems.

#### [A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/29956) (Chao et al., 2024)

- **Authors:** Chao, Wenshuo; Qiu, Zhaopeng; Wu, Likang; Guo, Zhuoning; Zheng, Zhi; Zhu, Hengshu; Liu, Hao
- **Year:** 2024
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper presents a Cross-view Hierarchical Graph Learning Hypernetwork (CHGH) for joint skill demand-supply prediction, addressing the limitations of existing methods that either rely on domain expertise or simplify skill evolution as a time-series problem. The CHGH framework consists of a cross-view graph encoder to capture demand-supply interconnections, a hierarchical graph encoder to model skill co-evolution, and a conditional hyper-decoder to predict variations in demand and supply. Experimental results on three real-world datasets show that CHGH outperforms seven baselines and effectively integrates these components for improved prediction accuracy.

#### [Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting](https://www.ijcai.org/proceedings/2024/222) (Chen et al., 2024)

- **Authors:** Chen, Xi; Qin, Chuan; Wang, Zhigaoyuan; Cheng, Yihang; Wang, Chao; Zhu, Hengshu; Xiong, Hui
- **Year:** 2024
- **Publication:** IJCAI 2024
- **Summary:** This work proposes a novel approach, Pre-DyGAE, for occupational skill demand forecasting by focusing on the dynamic nature of skill demand at an occupational level. The model integrates a graph autoencoder pre-trained with a semantically-aware and uncertainty-aware encoder-decoder structure, using contrastive learning to handle data sparsity and a unified loss for imbalanced distributions. The method is further enhanced with temporal encoding units and a shift module, achieving significant improvements in forecasting accuracy across four real-world datasets.

### Job Title Benchmarking

#### [Job2Vec: Job Title Benchmarking with Collective Multi-View Representation Learning](https://dl.acm.org/doi/10.1145/3357384.3357825) (Zhang et al., 2019)

- **Authors:** Zhang, Denghui; Liu, Junming; Zhu, Hengshu; Liu, Yanchi; Wang, Lichen; Wang, Pengyang; Xiong, Hui
- **Year:** 2019
- **Publication:** Proceedings of the 28th ACM International Conference on Information and Knowledge Management
- **Summary:** This paper presents Job2Vec, a novel approach to Job Title Benchmarking (JTB) that leverages a data-driven solution using a large-scale Job Title Benchmarking Graph (Job-Graph) constructed from extensive career records. The method addresses challenges like non-standard job titles, missing information, and limited job transition data by employing collective multi-view representation learning, examining graph topology, semantic meaning, transition balance, and transition duration. Extensive experiments demonstrate that the proposed method effectively predicts links in the Job-Graph, enabling accurate matching of similar-level job titles across companies.

### Salary Prediction

#### [Collaborative Company Profiling: Insights from an Employee's Perspective](https://ojs.aaai.org/index.php/AAAI/article/view/10751) (Lin et al., 2017)

- **Authors:** Lin, Hao; Zhu, Hengshu; Zuo, Yuan; Zhu, Chen; Wu, Junjie; Xiong, Hui
- **Year:** 2017
- **Publication:** Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence
- **Summary:** This proposes a novel method called Company Profiling based Collaborative Topic Regression (CPCTR) to develop company profiles from an employee's perspective using online ratings and comments. CPCTR employs a joint optimization framework to model both textual information, such as reviews, and numerical information, like salaries and ratings, enabling the identification of latent structural patterns and opinions. The method was validated through extensive experiments on real-world data and demonstrated a more comprehensive understanding of company characteristics and more effective salary prediction compared to other baseline methods. The longer version was published in TOIS 2020.

### Job Mobility Analysis

#### [Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting](https://doi.org/10.1145/3287560.3287572) (De-Arteaga et al., 2019)

- **Authors:** De-Arteaga, Maria; Romanov, Alexey; Wallach, Hanna; Chayes, Jennifer; Borgs, Christian; Chouldechova, Alexandra; Geyik, Sahin; Kenthapadi, Krishnaram; Kalai, Adam Tauman
- **Year:** 2019
- **Publication:** Proceedings of the Conference on Fairness, Accountability, and Transparency
- **Summary:** This paper explores the presence and impact of gender bias in machine learning models for occupation classification using online biographies. The study demonstrates how explicit gender indicators (e.g., names, pronouns) and implicit biases in semantic representations contribute to allocation harms, even after scrubbing explicit indicators. The findings reveal a correlation between gender disparities in true positive rates and occupational gender imbalances, highlighting the potential for machine learning to perpetuate existing inequalities.

#### [Preference-Constrained Career Path Optimization: An Exploration Space-Aware Stochastic Model](https://ieeexplore.ieee.org/abstract/document/10415838) (Guo et al., 2023)

- **Authors:** Guo, Pengzhan; Xiao, Keli; Zhu, Hengshu; Meng, Qingxin
- **Year:** 2023
- **Publication:** 2023 IEEE International Conference on Data Mining
- **Summary:** This study addresses the problem of optimizing career paths with user-defined preferences. The authors propose an exploration space-aware stochastic searching algorithm that integrates deep learning for determining search space and predicting position transitions. Through mathematical analysis and empirical validation on real-world datasets, the method is shown to outperform existing career path optimization approaches.

### Skill Demand Analysis

#### [A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/29956) (Chao et al., 2024)

- **Authors:** Chao, Wenshuo; Qiu, Zhaopeng; Wu, Likang; Guo, Zhuoning; Zheng, Zhi; Zhu, Hengshu; Liu, Hao
- **Year:** 2024
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This paper presents a Cross-view Hierarchical Graph Learning Hypernetwork (CHGH) for joint skill demand-supply prediction, addressing the limitations of existing methods that either rely on domain expertise or simplify skill evolution as a time-series problem. The CHGH framework consists of a cross-view graph encoder to capture demand-supply interconnections, a hierarchical graph encoder to model skill co-evolution, and a conditional hyper-decoder to predict variations in demand and supply. Experimental results on three real-world datasets show that CHGH outperforms seven baselines and effectively integrates these components for improved prediction accuracy.

#### [Pre-DyGAE: Pre-training Enhanced Dynamic Graph Autoencoder for Occupational Skill Demand Forecasting](https://www.ijcai.org/proceedings/2024/222) (Chen et al., 2024)

- **Authors:** Chen, Xi; Qin, Chuan; Wang, Zhigaoyuan; Cheng, Yihang; Wang, Chao; Zhu, Hengshu; Xiong, Hui
- **Year:** 2024
- **Publication:** IJCAI 2024
- **Summary:** This work proposes a novel approach, Pre-DyGAE, for occupational skill demand forecasting by focusing on the dynamic nature of skill demand at an occupational level. The model integrates a graph autoencoder pre-trained with a semantically-aware and uncertainty-aware encoder-decoder structure, using contrastive learning to handle data sparsity and a unified loss for imbalanced distributions. The method is further enhanced with temporal encoding units and a shift module, achieving significant improvements in forecasting accuracy across four real-world datasets.

### Skill Valuation

#### [Measuring the Popularity of Job Skills in Recruitment Market: A Multi-Criteria Approach](https://aaai.org/papers/11847-measuring-the-popularity-of-job-skills-in-recruitment-market-a-multi-criteria-approach/) (Xu et al., 2018)

- **Authors:** Xu, Tong; Zhu, Hengshu; Zhu, Chen; Li, Pan; Xiong, Hui
- **Year:** 2018
- **Publication:** Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence
- **Summary:** This paper introduces a data-driven approach to measure the popularity of job skills by analyzing large-scale recruitment data and constructing a job skill network. A novel Skill Popularity-based Topic Model (SPTM) is proposed, integrating diverse job criteria (e.g., salary levels, company size) and the latent relationships among skills to rank them based on multifaceted popularity. Experiments on real-world recruitment data demonstrate the effectiveness of SPTM, uncovering trends like popular skills associated with high-paying jobs.

#### [Analyzing the relationship between information technology jobs advertised on-line and skills requirements using association rules](https://beei.org/index.php/EEI/article/view/2590) (Patacsil et al., 2021)

- **Authors:** Patacsil, Frederick F.; Acosta, Michael
- **Year:** 2021
- **Publication:** Bulletin of Electrical Engineering and Informatics
- **Summary:** This paper proposes a method for analyzing the relationship between IT jobs and their required skills by utilizing job postings and applying association rule mining techniques. By using the FP-growth algorithm to discover frequent patterns and relationships in job postings, the study identifies how specific skills are linked to IT job requirements. The findings highlight gaps between educational training and industry demands, suggesting potential areas for curriculum development and policy changes by the Philippine government to better align with the labor market's needs.

### Hr Management

#### [May the bots be with you! Delivering HR cost-effectiveness and individualised employee experiences in an MNE](https://doi.org/10.1080/09585192.2020.1859582) (Malik et al., 2022)

- **Authors:** Malik, Ashish; Budhwar, Pawan; Patel, Charmi; Srikanth, N. R.
- **Year:** 2022
- **Publication:** The International Journal of Human Resource Management
- **Summary:** This paper explores the use of AI-enabled tools in HR management within a multinational enterprise (MNE) subsidiary in India. It shows that chatbots and digital assistants improve cost efficiency and create personalized employee experiences, leading to higher commitment and job satisfaction. The research highlights key implications for future HR practices and organizational strategies involving AI integration.

### Employee Evaluation

#### [Identifying High Potential Talent: A Neural Network Based Dynamic Social Profiling Approach](https://ieeexplore.ieee.org/document/8970676) (Ye et al., 2019)

- **Authors:** Ye, Yuyang; Zhu, Hengshu; Xu, Tong; Zhuang, Fuzhen; Yu, Runlong; Xiong, Hui
- **Year:** 2019
- **Publication:** 2019 IEEE International Conference on Data Mining (ICDM)
- **Summary:** This paper presents a GNN-based approach to identifying high-potential talent (HIPO) early in their careers by dynamically analyzing their behaviors in organizational social networks. It combines Graph Convolutional Networks (GCNs) for social profile modeling and adaptive Long Short Term Memory (LSTM) networks with a global attention mechanism to track and evaluate changes in employee social profiles over time. The approach reduces biases associated with subjective evaluations and demonstrates its effectiveness and interpretability through extensive real-world experiments.

### Employee Turnover Analysis

#### [Exploiting the Contagious Effect for Employee Turnover Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/3910) (Teng et al., 2019)

- **Authors:** Teng, Mingfei; Zhu, Hengshu; Liu, Chuanren; Zhu, Chen; Xiong, Hui
- **Year:** 2019
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This study proposes a novel approach for employee turnover prediction, integrating employee profiles, environmental factors, and the contagious effect of turnover behaviors among co-workers. The authors introduce a contagious effect heterogeneous neural network (CEHNN) and a global attention mechanism to evaluate the heterogeneous impact of turnover behaviors, enhancing the interpretability of predictions. Extensive experiments on a real-world dataset validate the effectiveness of incorporating the contagious effect in improving turnover prediction accuracy and providing actionable insights for talent retention.

### Organization Analysis

#### [Exploiting the Contagious Effect for Employee Turnover Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/3910) (Teng et al., 2019)

- **Authors:** Teng, Mingfei; Zhu, Hengshu; Liu, Chuanren; Zhu, Chen; Xiong, Hui
- **Year:** 2019
- **Publication:** Proceedings of the AAAI Conference on Artificial Intelligence
- **Summary:** This study proposes a novel approach for employee turnover prediction, integrating employee profiles, environmental factors, and the contagious effect of turnover behaviors among co-workers. The authors introduce a contagious effect heterogeneous neural network (CEHNN) and a global attention mechanism to evaluate the heterogeneous impact of turnover behaviors, enhancing the interpretability of predictions. Extensive experiments on a real-world dataset validate the effectiveness of incorporating the contagious effect in improving turnover prediction accuracy and providing actionable insights for talent retention.

#### [Exit Ripple Effects: Understanding the Disruption of Socialization Networks Following Employee Departures](https://dl.acm.org/doi/10.1145/3589334.3645634) (Gamba et al., 2024)

- **Authors:** Gamba, David; Yu, Yulin; Yuan, Yuan; Schoenebeck, Grant; Romero, Daniel M.
- **Year:** 2024
- **Publication:** WWW 2024
- **Summary:** This study explores the impact of employee departures on the socialization networks of remaining coworkers, using data from a large holding company. The study finds that communication tends to break down among the remaining employees, especially during periods of organizational stress, although some individuals may benefit from better network positioning after a departure. It highlights the need for organizations to consider both external and internal factors when managing workforce changes to maintain effective communication dynamics.

### Company Profiling

#### [Aspect-Sentiment Embeddings for Company Profiling and Employee Opinion Mining](https://doi.org/10.1007/978-3-031-23804-8_12) (Bajpai et al., 2019)

- **Authors:** Bajpai, Rajiv; Hazarika, Devamanyu; Singh, Kunal; Gorantla, Sruthi; Cambria, Erik; Zimmerman, Roger
- **Year:** 2019
- **Publication:** Proceedings of the 19th International Conference on Computational Linguistics and Intelligent Text Processing
- **Summary:** This work focuses on creating aspect-sentiment embeddings for companies by analyzing employee reviews, providing a novel way to evaluate organizations. Using Glassdoor reviews, the authors developed a dataset and employed an ensemble approach for aspect-level sentiment analysis, addressing a gap in similar work for companies. The proposed embeddings allow for personalized company profiling and provide insights to help individuals make informed decisions when choosing workplaces.

#### [Collaborative Company Profiling: Insights from an Employee's Perspective](https://ojs.aaai.org/index.php/AAAI/article/view/10751) (Lin et al., 2017)

- **Authors:** Lin, Hao; Zhu, Hengshu; Zuo, Yuan; Zhu, Chen; Wu, Junjie; Xiong, Hui
- **Year:** 2017
- **Publication:** Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence
- **Summary:** This proposes a novel method called Company Profiling based Collaborative Topic Regression (CPCTR) to develop company profiles from an employee's perspective using online ratings and comments. CPCTR employs a joint optimization framework to model both textual information, such as reviews, and numerical information, like salaries and ratings, enabling the identification of latent structural patterns and opinions. The method was validated through extensive experiments on real-world data and demonstrated a more comprehensive understanding of company characteristics and more effective salary prediction compared to other baseline methods. The longer version was published in TOIS 2020.


## Contact Information

- Naoki Otani <naoki [at] megagon.ai>


## Citation

```
@inproceedings{otani-etal-2025-natural,
    title = "Natural Language Processing for Human Resources: {A} Survey",
    author = "Otani, Naoki  and
      Bhutani, Nikita  and
      Hruschka, Estevam",
    booktitle = "Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Industry Track)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico, USA",
    publisher = "Association for Computational Linguistics"
}
```

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be BSD 3 clause license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.