# real
#  P   TP   FN
#  N   FP   TN
#      P    N
#     Model Prediction


def print_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision * recall)

    print (f"Accuracy:  {accuracy*100:.2f}%")
    print (f"Precision:  {precision*100:.2f}%")
    print (f"Recall:  {recall*100:.2f}%")
    print (f"F1:  {f1*100:.2f}%")

print ("50% de acertos proporcionais: ")
print_metrics(5, 45, 5, 45)

print ("20% de acertos proporcionais: ")
print_metrics(2, 12, 8, 78)