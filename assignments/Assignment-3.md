# Assignment-3: 18/June (Statistics for Machine Learning)

### Part 1
* **What have you learnt from the Titanic Disaster?** Survival was not random; it was heavily influenced by socioeconomic status and demographic factors (Gender and Age).
* **Distractions are dangerous, elaborate:** Irrelevant features (noise) distract the model, causing overfitting and poor predictions.
* **Stakeholders should be kept informed, yes/no:** **Yes**. Aligns the product with business requirements and builds trust.
* **Traceability is essential, what do you think?** **Yes**. Crucial for debugging, auditing for bias, and explaining outcomes.
* **Documentation may have lasting benefits, true or false and why:** **True**. Ensures future developers can understand, maintain, and reproduce your work.

### Part 2: Investigation Questions
1. **Key factors:** Gender, Ticket Class (Pclass), and Age.
2. **How did variables affect survival?** Females, 1st class, and younger passengers had significantly higher survival chances.
3. **Can we identify significant predictors?** Yes, using correlation matrices, Gender and Pclass are the strongest predictors.
4. **Are differences statistically significant?** Yes, hypothesis tests (like Chi-Square) yield p-values < 0.05.

### Part 3: Reflection
* **Which statistical measure gave the most insight?** Correlation coefficients and grouped Means.
* **How did hypothesis testing validate your assumptions?** Provided mathematical proof via low p-values that differences in survival were statistically significant.
* **How does regression connect statistics to ML prediction?** Regression uses statistical concepts (minimizing variance) to fit a mathematical function to data, which ML uses to predict outcomes on new data.
