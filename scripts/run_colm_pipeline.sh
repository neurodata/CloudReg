# exit when any command fails
set -e

while getopts i:o:c:e: option
do
case "${option}"
in
i) IBUCKET=${OPTARG};;
o) OBUCKET=${OPTARG};;
c) CHANNEL=${OPTARG};;
e) EXP=${OPTARG};;
esac
done

SSD1=/home/ubuntu/ssd1
SSD2=/home/ubuntu/ssd2
OUTDIR=${SSD1}

#sudo ./mount_ssds.sh

# enter virtualenv
#. /home/ubuntu/bias_correction/bin/activate
#
## compute and apply bias
#python ec2_compute_bias.py --in_bucket_path ${IBUCKET}VW0/ --bias_bucket_name ${OBUCKET} --channel ${CHANNEL} --experiment_name ${EXP} --outdir ${OUTDIR}/
#
## kill all remaining processes in case
#pkill -f python
#
## go into VW0 directory
cd ${OUTDIR}/VW0
#
## make directory for stitched data
#mkdir -p ${SSD2}/stitched_data
#
#aws s3 cp ${IBUCKET}Experiment.ini ${OUTDIR}/
#aws s3 cp ${IBUCKET}Scanned\ Cells.txt ${OUTDIR}/
#. /home/ubuntu/bias_correction/bin/activate
#python ~/generate_cobalt_preprocessing_commands.py --stitched_dir ${SSD2}/stitched_data --stack_dir ${OUTDIR}/VW0 --config_file ${OUTDIR}/Experiment.ini --scanned_cells  ${OUTDIR}/Scanned\ Cells.txt --channel ${CHANNEL}
#
## leave virtualenv for terastitcher
#deactivate
## run Terastitcher
#if [[ "$CHANNEL" != "0" ]]; then 
#	cp ~/xml_files/${EXP}/*.xml ./
#fi
#echo "******** Starting Terastitcher ********"
#bash terastitcher_commands.sh
#mkdir  -p ~/xml_files/${EXP}
#cp *.xml ~/xml_files/${EXP}/
#pkill -f python3

# enter virtualenv
. /home/ubuntu/bias_correction/bin/activate

# create precomputed volume
STITCHED_PATH=${SSD2}/stitched_data/$(ls $SSD2/stitched_data)
echo $STITCHED_PATH
python ~/create_precomputed_volume_v3.py ${STITCHED_PATH} ${OUTDIR}/VW0/xml_import.xml s3://${OBUCKET}/precomputed_volumes/${EXP}/CHN0${CHANNEL} 

#sudo shutdown now
