#!/bin/sh
# Requires java, bcftools, and tabix

BEAGLE5_JAR=$1
DATA_DIR=$2
GENETIC_MAP=$3
nref=$4 # number of sample in the reference panel
ntarget=$5 # number of imputation target

cd ${DATA_DIR}

# if [ -f sim.vcf.gz.pos ]; then
#   rm sim.vcf.gz.pos
# fi
# bcftools +fill-AN-AC sim.vcf.gz | bcftools query -f '%POS\t%AN\t%AC\n' > sim.vcf.gz.pos

# subset samples
bcftools query -l sim.vcf.gz > sim.vcf.gz.sample
head -n $nref sim.vcf.gz.sample > ref.sample
tail -n $ntarget sim.vcf.gz.sample > target.sample
bcftools view -S ref.sample sim.vcf.gz -Oz -o ref.vcf.gz
tabix -f ref.vcf.gz
bcftools view -S target.sample sim.vcf.gz -Oz -o target.vcf.gz
tabix -f target.vcf.gz

# create chip data
bcftools view -R sim.chip.variants target.vcf.gz -Oz -o target_chip.vcf.gz
tabix target_chip.vcf.gz

java -jar ${BEAGLE5_JAR} \
    map=${GENETIC_MAP} \
    ref=ref.vcf.gz \
    gt=target_chip.vcf.gz \
    out=imputed

tabix -f imputed.vcf.gz

# bcftools +fill-tags imputed.vcf.gz | bcftools query -f '%POS\t%AF\n' > imputed.vcf.gz.pos &
# bcftools +fill-tags target.vcf.gz | bcftools query -f '%POS\t%AF\n' > target.vcf.gz.pos &

# bcftools query -f '%POS\t%REF\t%ALT[\t%DS]\n' imputed.vcf.gz > imputed.tsv
bcftools query -f '%POS %REF %ALT[ %DS]\n' imputed.vcf.gz > imputed.csv # actually space-separated
