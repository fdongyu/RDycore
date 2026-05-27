#!/bin/sh

mach=
project_id=
rdycore_dir=
frontier_node_type=
N=1
is_petsc_run=0
is_cpu_run=0
is_gpu_run=0
problem_name=Turning_30m
no_ceed=0

display_help() {
  echo "Usage: $0 " >&2
  echo
  echo "   -h, --help                               Display this message"
  echo "   --rdycore-dir                            Path to RDycore directory"
  echo "   --mach <pm-cpu|pm-gpu|frontier>          Supported machine name"
  echo "   --frontier-node-type <cpu|gpu>           To run on Frontier CPUs or GPUs"
  echo "   -N --node  <N>                           Number of nodes (default = 1)"
  echo "   --project <project-id>                   Project ID that will charged for the job"
  echo "   --problem_name <Turning_30m|Houston1km>  Supported problems"
  echo "   --no_ceed                                Use PETSc-version of RDycore without CEED"
  return 0
}  


while [ $# -gt 0 ]
do
  case "$1" in
    --mach ) mach="$2"; shift ;;
    --project) project_id="$2"; shift ;;
    --frontier-node-type) frontier_node_type="$2"; shift ;;
    --rdycore-dir) rdycore_dir="$2"; shift ;;
    -N | --node) N="$2"; shift ;;
    --problem_name) problem_name="$2"; shift ;;
    --no_ceed) no_ceed=1; shift ;;
    -h | --help)
      display_help
      exit 0
      ;;
    -*)
      echo "Unsupported argument: $1"
      display_help
      exit 0
      ;;
    *)  break;;    # terminate while loop
  esac
  shift
done

# Check the machine command line argument
if [ "$mach" == "pm-cpu" ]; then
  # Determine the name of the batch file
  batch_filename=${problem_name}.${mach}.dam_break_sed.N_${N}.batch

  n=$((N*128))
  constraint="cpu"
  if [ $no_ceed -eq 1 ]; then
    is_petsc_run=1
  else
    is_cpu_run=1
  fi

elif [ "$mach" == "pm-gpu" ]; then
  # Determine the name of the batch file
  batch_filename=${problem_name}.${mach}.dam_break_sed.N_${N}.batch

  n=$((N*4))
  constraint="gpu"
  is_gpu_run=1

elif [ "$mach" == "frontier" ]; then

  # Make sure both CPU and GPU options were not specified
  if [ "$frontier_node_type" == "cpu" ]
  then
    # Determine the name of the batch file
    batch_filename=${problem_name}.${mach}-cpu.dam_break_sed.N_${N}.batch

    is_cpu_run=1
    n=$((N*56))
  elif [ "$frontier_node_type" == "gpu" ]
  then
    # Determine the name of the batch file
    batch_filename=${problem_name}.${mach}-gpu.dam_break_sed.N_${N}.batch

    is_gpu_run=1
    n=$((N*8))
  else
    echo "Unknown -frontier-node-type $frontier_node_type"
    display_help
    exit 0
  fi

elif [ "$mach" == "" ]; then
  echo "No machine specified via --mach"
  display_help
  exit 0
else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

# Make sure project was specified
if [ "$project_id" == "" ]
then
  echo "Project ID was not specified via --project"
  display_help
  exit 0
fi

if [ "$problem_name"  == "dam_break_sed" ]; then
  yaml_file=dam_break_sed.yaml
  exo_file=dam_break_sed_dx1.exo
  ic_file=dam_break_sed_dx1_IC.bin
  ic_file_sed=dam_break_sed_dx1_IC_sed.bin
else
  echo "Problem name is unsupported"
  display_help
  exit 0
fi

# Check if RDycore dir exists
source $rdycore_dir/config/set_petsc_settings.sh --mach $mach --config 1;

mpi_helper="$rdycore_dir/cmake/extract_mpi_include_directories.cmake"

if grep -q 'message(FATAL_ERROR "Could not find MPI include directories from PETSc' "$mpi_helper"; then
  perl -0pi -e 's/message\(FATAL_ERROR "Could not find MPI include directories from PETSc \(neither MPICXX_INCLUDES nor MPICC_SHOW available\)\."\)/message(WARNING "Could not find MPI include directories from PETSc (neither MPICXX_INCLUDES nor MPICC_SHOW available).")\n  set(${directories} "" PARENT_SCOPE)/' "$mpi_helper"
fi

if [ ! -d "$rdycore_dir" ]
then
  echo "The specified RDycore directory does not exist: $rdycore_dir"
  exit 0
else
  if [ ! -d "$rdycore_dir/build-$PETSC_ARCH" ]
  then
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "The following expected RDycore build directory not found:$rdycore_dir/build-$PETSC_ARCH "
    echo "So, attempting to build RDycore."
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    curr_dir=$PWD
    cd $rdycore_dir
    cmake -S . -B build-$PETSC_ARCH -DCMAKE_INSTALL_PREFIX=$PWD/build-$PETSC_ARCH -G Ninja
    cd build-$PETSC_ARCH
    ninja -j4 install
    cd $curr_dir

  fi
fi

if [ "$mach" == "pm-cpu" ] || [ "$mach" == "pm-gpu" ]
then

  # set problem-specific data dir
  if [ "$problem_name" == "dam_break_sed" ]; then
    data_dir=/global/cfs/projectdirs/m4267/dongyu/DATA/dam_break_sed/mesh
  else
    echo "Set data directory for problem"
    exiit 0
  fi

  if [ ! -f $exo_file ]
  then
    ln -s $data_dir/$exo_file .
  fi

  if [ ! -f $ic_file ]
  then
    ln -s $data_dir/$ic_file .
  fi

  if [ ! -f $ic_file_sed ]
  then
    ln -s $data_dir/$ic_file_sed .
  fi


  cp -f perlmutter.dam_break_sed.batch.base ${batch_filename}

  sed -i "s/PLACEHOLDER_RDYCORE_DIR/${rdycore_dir//\//\\/}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_MACHINE_NAME/${mach}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_N/${N}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_n/${n}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_PROJECT_ID/${project_id}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_CONSTRAINT/${constraint}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_PETSC_RUN/${is_petsc_run}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_CPU_RUN/${is_cpu_run}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_GPU_RUN/${is_gpu_run}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_YAML_FILE/${yaml_file}/g" ${batch_filename}

elif [ "$mach" == "frontier" ]
then
  if [ ! -f Turning_30m_with_z.updated.with_sidesets.exo ]
  then
    ln -s /lustre/orion/cli192/proj-shared/data/harvey/Turning_30m/Turning_30m_with_z.updated.with_sidesets.exo
  fi

  if [ ! -f solution_219.int64.dat ]
  then
    ln -s /lustre/orion/cli192/proj-shared/data/harvey/Turning_30m/solution_219.int64.dat
  fi

  cp -f frontier.critical_outflow_bc.batch.base ${batch_filename}

  sed -i "s/PLACEHOLDER_RDYCORE_DIR/${rdycore_dir//\//\\/}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_MACHINE_NAME/${mach}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_N/${N}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_n/${n}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_PROJECT_ID/${project_id}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_CPU_RUN/${is_cpu_run}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_GPU_RUN/${is_gpu_run}/g" ${batch_filename}
  sed -i "s/PLACEHOLDER_PYAML_FILE/${yaml_file}/g" ${batch_filename}

fi

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Created the ${batch_filename} that can be submitted via:  sbatch ${batch_filename}"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

