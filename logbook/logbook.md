# Logbook

## Meeting Log - June 3, 2025

**Date:** June 3, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Raul Adriaensen, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Based on previously collected information, proposed an approach to predict future coastline changes driven by sea level rise and other factors.

**Feedback Received:**  
1. In addition to sea level rise, multiple other factors (e.g., tides, monsoons) significantly influence coastal and island morphology and should be considered.  
2. It is necessary to read more relevant literature to justify the novelty and feasibility of the proposed methodology.  
3. Consider leveraging existing weather prediction models to support the development of the forecasting framework.

**Planned Actions Before Next Meeting:**  
1. Conduct an extensive review of relevant literature to support the proposed methodology.  
2. Complete the first draft of the IRP project plan.

---

## Meeting Log - June 10, 2025

**Date:** June 10, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Raul Adriaensen, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Presented the full IRP project plan for feedback and clarification on overall workflow and methodology.

**Feedback Received:**  
1. Reconsider the input parameters of the model, particularly from the perspective of coastal transects.  
2. Review more literature to critically assess the novelty and validity of the proposed resilience assessment approach.  
3. Reflect on whether the proposed workload is feasible within the three-month timeline. Prioritize identifying which tasks are core to the project and which are supplementary.

**Planned Actions Before Next Meeting:**  
1. Finalize and submit the complete IRP project plan.  
2. Begin initial experiments and preliminary work.

## Meeting Log - June 18, 2025

**Date:** June 18, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Myriam Prasow-Émond, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Discussed shifting the study site within the Maldives from Malé and Hulhumalé to Meedhoo, and examined the necessity and validity of this change. Also considered strategies for constructing a large-scale time series dataset to support future modeling.

**Feedback Received:**  
1. Use existing atoll data, particularly the six islands from Kench’s 2016 study, as initial analysis targets.  
2. Explore incorporating high-resolution UK CP18 climate scenarios used by the Singapore team into the analysis.  
3. Consider building a regression model to predict island morphology, as an alternative to traditional time series modeling, to evaluate its potential.

**Planned Actions Before Next Meeting:**  
1. Complete data extraction from island morphology datasets, including tidal correction and initial shoreline preprocessing.  
2. Study the details of the feedback provided in this meeting.

## Meeting Log - June 25, 2025

**Date:** June 25, 2025  
**Attendees:** Yves Plancherel, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Discussed detailed experimental results and analyses for three representative islands out of the eight target islands. Also reviewed the use of shoreline transects and the application of the FES2022 global tidal model for tidal correction.

**Feedback Received:**  
This part has been completed well. For the next stage of model development, consider incorporating island morphology as an input feature. Refer to *Statistics and Data Analysis in Geology* by John Davis for potential methodologies.

**Planned Actions Before Next Meeting:**  
1. Finalize dataset integration.  
2. Begin initial model construction using one natural island, consulting relevant literature and resources during the process.

## Meeting Log - July 1, 2025

**Date:** July 1, 2025  
**Attendees:** Matthew Piggott, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Shared reflections on dataset preparation. Discussed the feasibility of using the Radial Transect method and debated whether the project should focus on shoreline *prediction* or *projection*. Also outlined expectations for the upcoming modeling phase.

**Feedback Received:**  
Encouraged to continue exploring modeling approaches.

**Planned Actions Before Next Meeting:**  
1. Review relevant literature.  
2. Complete preliminary machine learning modeling exploration on the shoreline dataset.

## Meeting Log - July 8, 2025

**Date:** July 8, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Raul Adriaensen, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Analyzed experimental results of eight machine learning models, identifying XGBoost as the best-performing one. Evaluated feature importance and examined results and conclusions from other islands under these models.

**Feedback Received:**  
1. Clarify whether the project aims to build a model or propose a transferable methodology. Ensure the research contributes beyond a single island.  
2. Investigate what "good" and "poor" model performance specifically mean in the context of shoreline prediction, and visualize the outcomes to support analysis.

**Planned Actions Before Next Meeting:**  
1. Test model applicability on similar islands and evaluate transferability.  
2. Visualize and analyze model prediction performance.  
3. Reflect on and assess the limitations and strengths of the current analytical methods.

## Meeting Log - July 15, 2025

**Date:** July 15, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Discussed the performance of data from three reef islands across seven machine learning models. Proposed the hypothesis of “cross-island generalization” and considered the possibility of constructing a physics-informed model that performs well even on unseen islands. Debated the constraints imposed by the current radial transect framework on training features. Investigated possible reasons for one island's extremely poor model performance.

**Feedback Received:**  
Consider whether to use the radial transect framework by conducting more experiments to demonstrate its advantages and disadvantages, as well as its influence on feature extraction. Suggested trying a time-based data split to differentiate training and prediction sets.

**Planned Actions Before Next Meeting:**  
1. Train models on more islands to assess whether geographic factors explain the poor performance on the outlier island.  
2. Experiment with time-based splitting between training and prediction datasets.  
3. Provide well-supported, experiment-based reasoning for using the radial transect framework instead of traditional equidistant transects aligned parallel and perpendicular to shorelines.

## Meeting Log - July 22, 2025

**Date:** July 22, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Raul Adriaensen, Yiyu Yang, Jianing Wu

**Key Points Discussed:**  
Expanded the dataset from three to seven reef islands, each with nine years of data. Concluded that the previously identified outlier island had undergone recent artificial coastal construction. Presented solid, experiment-based justification for the necessity of the radial transect framework. Discussed the importance and necessity of reef-related parameters.

**Feedback Received:**  
Try applying the training method to islands in the Pacific. Consider a hybrid approach combining radial and traditional transect frameworks. Begin outlining the final project report.

**Planned Actions Before Next Meeting:**  
1. Test the model on selected Pacific islands.  
2. Explore a hybrid model using both radial and traditional transect frameworks.  
3. Design and implement comprehensive experiments to test the feasibility of cross-island generalization.

## Meeting Log - July 29, 2025

**Date:** July 29, 2025  
**Attendees:** Yves Plancherel, Matthew Piggott, Yiyu Yang, Jianing Wu  

**Key Points Discussed:**  
Focused on the issue of model transferability. Discussed whether the poor performance of the model on completely unseen islands is due to limited learning ability or potential shortcomings in the radial transect framework’s ability to reasonably explain island shapes.  

**Feedback Received:**  
Consider and clarify how the radial transect framework explains island shapes compared to the traditional perpendicular-to-shoreline transect approach, and assess its necessity.  

**Planned Actions Before Next Meeting:**  
1. Conduct experiments using the traditional perpendicular-to-shoreline transect method.  
2. Organize code and experimental logic, and begin drafting the outline of the final report.  

