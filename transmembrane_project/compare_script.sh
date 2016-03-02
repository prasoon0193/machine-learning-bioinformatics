for i in 0 1 2 3 4 5 6 7 8 9
do
  python compare_tm_pred.py Dataset160/set160.$i.labels.txt Dataset160/set160.$i.labels_PREDICTIONS.txt 
  printf "\n\n\n"
done