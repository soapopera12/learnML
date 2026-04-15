# Performance Metrics

---

## Accuracy

Accuracy measures the proportion of correct predictions out of total predictions.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

## Precision

Precision measures the proportion of correctly predicted positive observations out of all predicted positives.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

## Recall (Sensitivity)

Recall measures the proportion of actual positives correctly identified.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

## F1 Score

F1 Score is the harmonic mean of precision and recall.

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## False Positive Rate (FPR)

False Positive Rate measures the proportion of actual negative cases that are incorrectly classified as positive.

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

---

## ROC-AUC

ROC-AUC measures the ability of a model to distinguish between classes across different thresholds.

$$
\text{AUC} = \int_0^1 TPR \, d(FPR)
$$

---

## PR-AUC (Precision-Recall AUC)

PR-AUC measures the trade-off between precision and recall across different classification thresholds.

$$
\text{PR-AUC} = \int_0^1 \text{Precision} \, d(\text{Recall})
$$

---

## Confusion Matrix

A confusion matrix summarizes prediction results.

Predicted Positive | Predicted Negative 
|---|---|
Actual Positive | TP & FN 
Actual Negative | FP & TN 