# Assignment-6: 23/June (Logistic Regression Mini Quiz)

* **Question 1:** A passenger has a predicted probability $P = 0.62$. 
  * **Classification:** Survived. 
  * **Why the 0.5 threshold matters:** It's the standard decision boundary. Shifting it alters the balance between false positives and false negatives.
* **Question 2:** A passenger has $P = 0.38$. 
  * **What it means:** 38% estimated probability of survival.
  * **Threshold change:** If the threshold was lowered to 0.4, $0.38 < 0.4$, so the decision is still "Did Not Survive".
* **Question 3:** A passenger has $P = 0.85$. 
  * **Interpret:** Very high likelihood (85%) of survival. 
  * **Coefficients:** Age likely has a negative coefficient. Class likely has a negative coefficient (if classes are encoded 1, 2, 3).
* **Bonus Challenge:** If "female" has a positive coefficient, being female increases the log-odds (and therefore probability) of survival.
