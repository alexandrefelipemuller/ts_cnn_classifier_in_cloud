#awk 'BEGIN{FS=OFS=","} {s=$NF; for (i=NF-1; i>=1; i--) s = s OFS $i; print s}' FordA_TRAIN.arff > FordA_TRAIN
#awk 'BEGIN{FS=OFS=","} {s=$NF; for (i=NF-1; i>=1; i--) s = s OFS $i; print s}' FordA_TEST.arff > FordA_TEST


alias mc=~/machine_learning/minio/mc 
for i in $(mc ls myminio/mybucket| cut -d' ' -f6); do mc rm myminio/mybucket/$i; done

CHUNKS=5

num_base=$(cat FordA_TRAIN| wc -l)
part_size=$((num_base/"$CHUNKS"))

sort -R FordA_TRAIN > ATRAIN
for i in `seq "$CHUNKS"`; do
	cat ATRAIN |  head -n$((part_size*i)) | tail -n$part_size > ../"$i"/"$i"_TRAIN ;
done

rm ATRAIN


num_base=$(cat FordA_TEST| wc -l)
part_size=$((num_base/"$CHUNKS"))

sort -R FordA_TEST > ATEST
for i in `seq "$CHUNKS"`; do 
	cat ATEST |  head -n$((part_size*i)) | tail -n$part_size > ../"$i"/"$i"_TEST;
done
rm ATEST
