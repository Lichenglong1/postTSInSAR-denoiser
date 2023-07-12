#!/bin/bash 

conda activate insar
dir=
cd $dir
mkdir raw dem aux ifgs_result mintpy

###1.download dem
cd dem
dem.py -a stitch -b   -r -s 1 -c -u https://step.esa.int/auxdata/dem/SRTMGL1/  

###2.download S1 image
cd ..
cd raw 
python "./SSARA/ssara_federated_query.py" -p SENTINEL-1A,SENTINEL-1B -r 142 -b ' ' -s   -e   --kml
python "./SSARA/ssara_federated_query.py" -p SENTINEL-1A,SENTINEL-1B -r 142 -b ' ' -s   -e   --download --parallel=6


###3.download orbits
supermaster=
dem_file=

stackSentinel.py -s ./raw/ -d ./dem/$dem_file -a ./aux/ -o ./orbits/ -b ' ' -m $supermaster -C geometry -c 12 

cp -r $dir/orbits/*/* $dir/orbits
rm -rf $dir/configs  $dir/run_files

stackSentinel.py -s ./raw/ -d ./dem/$dem_file -a ./aux/ -o ./orbits/ -b '' -m $supermaster -C geometry -c 12

###4. generating run files
cd $dir/run_files
function ergodic()
{
           for file in ` ls $1`
             do
                 if [ -d $1"/"$file ]
                 then
                       ergodic $1"/"$file
                 else
                       local path=$1"/"$file 
                       local name=$file       
                    echo $1"/"$file >> slc_filelist.txt
                                    
                   fi
            done
}
                INIT_PATH="$dir"
                ergodic $INIT_PATH
find $INIT_PATH
grep run slc_filelist.txt >> run.txt 
rm -r slc_filelist.txt

cd $dir
for ((i=1; i<=11; i++)); do   
line_master=$i
run_step=`awk 'NR=='$line_master' {print $1}' $dir/run_files/run.txt`
bash $run_step
done


###5.time-series analysis using mintpy
conda activate insar
dir=
cd $dir/mintpy
smallbaselineApp.py $dir/mintpy/inputs/minty_run_timeseries_analysis.txt 





