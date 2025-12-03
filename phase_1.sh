
pth="/homes/bhavana/rat-scoring/batch/videos/NO_TMS/"
project_name="projectX"

if [ ! -d "./${project_name}" ]; then
  mkdir -p "$project_name"
  mkdir -p "./${project_name}/results/"
  mkdir -p "./${project_name}/out_files/"
  mkdir -p "./${project_name}/first_frames/"
  echo "Directory '$project_name' setup."
else
  echo "Directory '$project_name' already exists and files will be overwritten"
fi

for file in $pth*.mp4; do
		if [[ "$file" == *"$isolate"* ]]; then
			prefix=${file##*/}
                        prefix=${prefix::-4}
                        #echo "$file" "$prefix" "${step}_runner.sh"
			sbatch --job-name=$prefix-1 --output=./$project_name/out_files/$prefix-1.out 1_runner.sh $file $project_name
                        sbatch --job-name=$prefix-2 --output=./$project_name/out_files/$prefix-2.out 2_runner.sh $file $project_name
                fi
done
