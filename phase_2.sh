
pth="/homes/bhavana/rat-scoring/batch/videos/NO_TMS/"
project_name="projectX"
for file in $pth*.mp4; do
		if [[ "$file" == *"$isolate"* ]]; then
			prefix=${file##*/}
                        prefix=${prefix::-4}
                        #echo "$file" "$prefix" "${step}_runner.sh"
			sbatch --job-name=$prefix-3 --output=./$project_name/out_files/$prefix-3.out 3_runner.sh $file $project_name
                        sbatch --job-name=$prefix-4 --output=./$project_name/out_files/$prefix-4.out 4_runner.sh $file $project_name
                fi
done
